import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class Generator(nn.Module):
    """
    Generator network for Conditional GAN.
    Takes noise vector + condition vector as input and generates synthetic quasi-identifiers.
    """
    
    def __init__(self, noise_dim, condition_dim, target_dim, hidden_dims=[128, 256, 128]):
        """
        Initialize the Generator.
        
        Parameters:
        -----------
        noise_dim : int
            Dimension of noise vector
        condition_dim : int
            Dimension of condition vector (critical features)
        target_dim : int
            Dimension of target vector (quasi-identifiers to generate)
        hidden_dims : list
            List of hidden layer dimensions
        """
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.target_dim = target_dim
        
        # Input layer: noise + condition
        input_dim = noise_dim + condition_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, target_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, noise, condition):
        """
        Forward pass through the generator.
        
        Parameters:
        -----------
        noise : torch.Tensor
            Random noise vector
        condition : torch.Tensor
            Condition vector (critical features)
            
        Returns:
        --------
        torch.Tensor
            Generated synthetic quasi-identifiers
        """
        # Concatenate noise and condition
        x = torch.cat([noise, condition], dim=1)
        return self.network(x)

class Discriminator(nn.Module):
    """
    Discriminator network for Conditional GAN.
    Takes real/fake quasi-identifiers + condition vector as input and outputs probability.
    """
    
    def __init__(self, target_dim, condition_dim, hidden_dims=[128, 256, 128]):
        """
        Initialize the Discriminator.
        
        Parameters:
        -----------
        target_dim : int
            Dimension of target vector (quasi-identifiers)
        condition_dim : int
            Dimension of condition vector (critical features)
        hidden_dims : list
            List of hidden layer dimensions
        """
        super(Discriminator, self).__init__()
        
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        
        # Input layer: target + condition
        input_dim = target_dim + condition_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output probability [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, target, condition):
        """
        Forward pass through the discriminator.
        
        Parameters:
        -----------
        target : torch.Tensor
            Target vector (real or fake quasi-identifiers)
        condition : torch.Tensor
            Condition vector (critical features)
            
        Returns:
        --------
        torch.Tensor
            Probability that input is real
        """
        # Concatenate target and condition
        x = torch.cat([target, condition], dim=1)
        return self.network(x)

class ConditionalGAN:
    """
    Conditional GAN for synthetic data generation.
    """
    
    def __init__(self, noise_dim=100, device='auto'):
        """
        Initialize the Conditional GAN.
        
        Parameters:
        -----------
        noise_dim : int
            Dimension of noise vector
        device : str
            Device to use ('auto', 'cpu', 'cuda')
        """
        self.noise_dim = noise_dim
        self.device = self._get_device(device)
        
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        
        self.g_losses = []
        self.d_losses = []
        
        print(f"Using device: {self.device}")
    
    def _get_device(self, device):
        """Determine the best device to use."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def build_models(self, condition_dim, target_dim, hidden_dims=[128, 256, 128]):
        """
        Build generator and discriminator models.
        
        Parameters:
        -----------
        condition_dim : int
            Dimension of condition vector
        target_dim : int
            Dimension of target vector
        hidden_dims : list
            List of hidden layer dimensions
        """
        print(f"Building CGAN models...")
        print(f"  Condition dimension: {condition_dim}")
        print(f"  Target dimension: {target_dim}")
        print(f"  Hidden dimensions: {hidden_dims}")
        
        # Build generator
        self.generator = Generator(
            noise_dim=self.noise_dim,
            condition_dim=condition_dim,
            target_dim=target_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Build discriminator
        self.discriminator = Discriminator(
            target_dim=target_dim,
            condition_dim=condition_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        print("Models built successfully!")
    
    def train(self, condition_features, target_features, epochs=100, batch_size=32, 
              d_steps=1, g_steps=1, save_interval=10):
        """
        Train the Conditional GAN.
        
        Parameters:
        -----------
        condition_features : np.ndarray
            Condition features (critical features to preserve)
        target_features : np.ndarray
            Target features (quasi-identifiers to generate)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        d_steps : int
            Number of discriminator training steps per epoch
        g_steps : int
            Number of generator training steps per epoch
        save_interval : int
            Save models every N epochs
        """
        print(f"Starting CGAN training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  D steps per epoch: {d_steps}")
        print(f"  G steps per epoch: {g_steps}")
        
        # Convert to tensors
        condition_tensor = torch.FloatTensor(condition_features).to(self.device)
        target_tensor = torch.FloatTensor(target_features).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(condition_tensor, target_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            
            for batch_idx, (condition_batch, target_batch) in enumerate(dataloader):
                batch_size_actual = condition_batch.size(0)
                
                # Create labels
                real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)
                
                # Train Discriminator
                for _ in range(d_steps):
                    self.d_optimizer.zero_grad()
                    
                    # Real data
                    real_output = self.discriminator(target_batch, condition_batch)
                    d_real_loss = self.criterion(real_output, real_labels)
                    
                    # Fake data
                    noise = torch.randn(batch_size_actual, self.noise_dim).to(self.device)
                    fake_target = self.generator(noise, condition_batch)
                    fake_output = self.discriminator(fake_target.detach(), condition_batch)
                    d_fake_loss = self.criterion(fake_output, fake_labels)
                    
                    # Total discriminator loss
                    d_loss = d_real_loss + d_fake_loss
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    epoch_d_loss += d_loss.item()
                
                # Train Generator
                for _ in range(g_steps):
                    self.g_optimizer.zero_grad()
                    
                    # Generate fake data
                    noise = torch.randn(batch_size_actual, self.noise_dim).to(self.device)
                    fake_target = self.generator(noise, condition_batch)
                    fake_output = self.discriminator(fake_target, condition_batch)
                    
                    # Generator loss
                    g_loss = self.criterion(fake_output, real_labels)
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    epoch_g_loss += g_loss.item()
            
            # Calculate average losses
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_g_loss = epoch_g_loss / len(dataloader)
            
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
            
            # Save models periodically
            if (epoch + 1) % save_interval == 0:
                self.save_models(f"models/cgan_epoch_{epoch+1}")
        
        print("Training completed!")
    
    def generate_synthetic_data(self, condition_features, num_samples=None):
        """
        Generate synthetic quasi-identifiers using the trained generator.
        
        Parameters:
        -----------
        condition_features : np.ndarray
            Condition features (critical features to preserve)
        num_samples : int, optional
            Number of samples to generate (default: same as input)
            
        Returns:
        --------
        np.ndarray
            Generated synthetic quasi-identifiers
        """
        if self.generator is None:
            raise ValueError("Generator not trained. Call train() first.")
        
        if num_samples is None:
            num_samples = len(condition_features)
        
        print(f"Generating {num_samples} synthetic samples...")
        
        # Convert to tensor
        condition_tensor = torch.FloatTensor(condition_features[:num_samples]).to(self.device)
        
        # Generate noise
        noise = torch.randn(num_samples, self.noise_dim).to(self.device)
        
        # Generate synthetic data
        with torch.no_grad():
            synthetic_target = self.generator(noise, condition_tensor)
        
        # Convert back to numpy
        synthetic_data = synthetic_target.cpu().numpy()
        
        print(f"Synthetic data generated: {synthetic_data.shape}")
        return synthetic_data
    
    def save_models(self, save_dir):
        """Save generator and discriminator models."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.generator.state_dict(), f"{save_dir}/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{save_dir}/discriminator.pth")
        
        # Save training history
        history = {
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }
        with open(f"{save_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir):
        """Load generator and discriminator models."""
        if self.generator is None or self.discriminator is None:
            raise ValueError("Models not built. Call build_models() first.")
        
        self.generator.load_state_dict(torch.load(f"{load_dir}/generator.pth", map_location=self.device))
        self.discriminator.load_state_dict(torch.load(f"{load_dir}/discriminator.pth", map_location=self.device))
        
        # Load training history
        try:
            with open(f"{load_dir}/training_history.json", 'r') as f:
                history = json.load(f)
                self.g_losses = history['g_losses']
                self.d_losses = history['d_losses']
        except:
            print("Could not load training history")
        
        print(f"Models loaded from {load_dir}")
    
    def plot_training_history(self, save_path=None):
        """Plot training loss history."""
        plt.figure(figsize=(12, 5))
        
        # Generator loss
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses, label='Generator Loss', color='blue')
        plt.title('Generator Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Discriminator loss
        plt.subplot(1, 2, 2)
        plt.plot(self.d_losses, label='Discriminator Loss', color='red')
        plt.title('Discriminator Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def evaluate_model(self, real_target, synthetic_target):
        """
        Evaluate the quality of synthetic data.
        
        Parameters:
        -----------
        real_target : np.ndarray
            Real target features
        synthetic_target : np.ndarray
            Synthetic target features
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        print("Evaluating synthetic data quality...")
        
        metrics = {}
        
        # Basic statistics
        metrics['real_mean'] = np.mean(real_target, axis=0)
        metrics['real_std'] = np.std(real_target, axis=0)
        metrics['synthetic_mean'] = np.mean(synthetic_target, axis=0)
        metrics['synthetic_std'] = np.std(synthetic_target, axis=0)
        
        # Mean absolute error
        metrics['mae'] = np.mean(np.abs(real_target - synthetic_target))
        
        # Correlation preservation
        real_corr = np.corrcoef(real_target.T)
        synthetic_corr = np.corrcoef(synthetic_target.T)
        metrics['correlation_mae'] = np.mean(np.abs(real_corr - synthetic_corr))
        
        print(f"Evaluation completed:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Correlation MAE: {metrics['correlation_mae']:.4f}")
        
        return metrics
