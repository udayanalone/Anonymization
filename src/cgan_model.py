import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import re


class CGAN:
    """
    Conditional Generative Adversarial Network (CGAN) for generating synthetic data
    based on sensitive attributes.
    """
    
    def __init__(self, latent_dim=100, condition_dim=None):
        """
        Initialize the CGAN model.
        
        Parameters:
        -----------
        latent_dim : int, default=100
            Dimension of the latent space
        condition_dim : int, optional
            Dimension of the condition vector
        """
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.generator = None
        self.discriminator = None
        self.cgan = None
        self.data_shape = None
        self.scaler = MinMaxScaler()
        self.condition_encoder = None
        self.attribute_type = None
        
    def build_generator(self, data_shape):
        """
        Build the generator model.
        
        Parameters:
        -----------
        data_shape : int
            Shape of the data to generate
        """
        # Input for latent vector
        latent_input = layers.Input(shape=(self.latent_dim,))
        
        # Input for condition
        condition_input = layers.Input(shape=(self.condition_dim,))
        
        # Concatenate latent vector and condition
        x = layers.Concatenate()([latent_input, condition_input])
        
        # Hidden layers
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        output = layers.Dense(data_shape, activation='tanh')(x)
        
        # Define the model
        model = models.Model([latent_input, condition_input], output, name='generator')
        return model
    
    def build_discriminator(self, data_shape):
        """
        Build the discriminator model.
        
        Parameters:
        -----------
        data_shape : int
            Shape of the data to discriminate
        """
        # Input for data
        data_input = layers.Input(shape=(data_shape,))
        
        # Input for condition
        condition_input = layers.Input(shape=(self.condition_dim,))
        
        # Concatenate data and condition
        x = layers.Concatenate()([data_input, condition_input])
        
        # Hidden layers
        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Define the model
        model = models.Model([data_input, condition_input], output, name='discriminator')
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
        return model
    
    def build_cgan(self, generator, discriminator):
        """
        Build the CGAN model.
        
        Parameters:
        -----------
        generator : tf.keras.Model
            Generator model
        discriminator : tf.keras.Model
            Discriminator model
        """
        # For the combined model, we only train the generator
        discriminator.trainable = False
        
        # Input for latent vector
        latent_input = layers.Input(shape=(self.latent_dim,))
        
        # Input for condition
        condition_input = layers.Input(shape=(self.condition_dim,))
        
        # Generate data
        generated_data = generator([latent_input, condition_input])
        
        # Discriminate generated data
        validity = discriminator([generated_data, condition_input])
        
        # Define the combined model
        model = models.Model([latent_input, condition_input], validity, name='cgan')
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
        return model
    
    def preprocess_data(self, data, attribute_type):
        """
        Preprocess data for training.
        
        Parameters:
        -----------
        data : pandas.Series
            Data to preprocess
        attribute_type : str
            Type of the attribute ('name', 'ssn', 'email', etc.)
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed data
        """
        self.attribute_type = attribute_type
        
        # Convert to string if not already
        data = data.astype(str)
        
        # Different preprocessing based on attribute type
        if attribute_type == 'name':
            # Extract features like name length, number of words, etc.
            name_length = data.str.len()
            word_count = data.str.split().str.len()
            has_middle_name = data.str.split().str.len() > 2
            
            # Combine features
            features = pd.DataFrame({
                'name_length': name_length,
                'word_count': word_count,
                'has_middle_name': has_middle_name.astype(int)
            })
            
        elif attribute_type == 'ssn':
            # Extract parts of SSN
            ssn_parts = data.str.extract(r'(\d{3})[-]?(\d{2})[-]?(\d{4})')
            features = ssn_parts.fillna('000').astype(int)
            
        elif attribute_type == 'email':
            # Extract domain and username length
            domain = data.str.extract(r'@([\w.-]+)')
            username_length = data.str.split('@').str[0].str.len()
            
            # One-hot encode domains (limit to top 10 domains)
            top_domains = domain.value_counts().nlargest(10).index
            domain_dummies = pd.get_dummies(domain[0].apply(lambda x: x if x in top_domains else 'other'))
            
            # Combine features
            features = pd.concat([domain_dummies, pd.DataFrame({'username_length': username_length})], axis=1)
            
        elif attribute_type == 'phone':
            # Extract area code
            area_code = data.str.extract(r'\(?([0-9]{3})\)?[-\s]?')
            features = area_code.fillna('000').astype(int)
            
        elif attribute_type == 'address':
            # Extract features like address length, has street number, etc.
            address_length = data.str.len()
            has_street_number = data.str.contains(r'^\d+').astype(int)
            has_apt_number = data.str.contains(r'apt|suite|#', case=False).astype(int)
            
            # Combine features
            features = pd.DataFrame({
                'address_length': address_length,
                'has_street_number': has_street_number,
                'has_apt_number': has_apt_number
            })
            
        elif attribute_type == 'birthdate':
            # Extract year, month, day
            date_parts = data.str.extract(r'(\d{4})[-/]?(\d{1,2})[-/]?(\d{1,2})|([\d]{1,2})[-/]?([\d]{1,2})[-/]?([\d]{2,4})')
            
            # Handle different date formats
            if date_parts.iloc[:, 0].notna().sum() > date_parts.iloc[:, 3].notna().sum():
                # Format: YYYY-MM-DD
                year = date_parts.iloc[:, 0].fillna('1970').astype(int)
                month = date_parts.iloc[:, 1].fillna('1').astype(int)
                day = date_parts.iloc[:, 2].fillna('1').astype(int)
            else:
                # Format: MM-DD-YYYY or DD-MM-YYYY
                year = date_parts.iloc[:, 5].fillna('1970')
                # Fix 2-digit years
                year = year.apply(lambda x: '19' + x if len(str(x)) == 2 else x).astype(int)
                month = date_parts.iloc[:, 3].fillna('1').astype(int)
                day = date_parts.iloc[:, 4].fillna('1').astype(int)
            
            # Combine features
            features = pd.DataFrame({
                'year': year,
                'month': month,
                'day': day
            })
            
        elif attribute_type == 'credit_card':
            # Extract first 6 digits (BIN/IIN) which identify card type and issuer
            bin_iin = data.str.replace('-', '').str[:6]
            features = bin_iin.fillna('000000').astype(int)
            
        else:
            # For unrecognized types, use the data as is
            features = pd.DataFrame({'value': data})
        
        # Scale numerical features
        numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns
        if not numerical_cols.empty:
            features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        
        return features.values
    
    def preprocess_conditions(self, conditions):
        """
        Preprocess conditions for training.
        
        Parameters:
        -----------
        conditions : pandas.Series or numpy.ndarray
            Conditions to preprocess
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed conditions
        """
        if self.condition_encoder is None:
            self.condition_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            conditions_reshaped = np.array(conditions).reshape(-1, 1)
            self.condition_encoder.fit(conditions_reshaped)
        
        conditions_reshaped = np.array(conditions).reshape(-1, 1)
        return self.condition_encoder.transform(conditions_reshaped)
    
    def fit(self, data, conditions, epochs=2000, batch_size=32, sample_interval=200):
        """
        Train the CGAN model.
        
        Parameters:
        -----------
        data : pandas.Series
            Data to train on
        conditions : pandas.Series or numpy.ndarray
            Conditions for the data
        epochs : int, default=2000
            Number of epochs to train for
        batch_size : int, default=32
            Batch size for training
        sample_interval : int, default=200
            Interval to sample and save generated data
        """
        # Preprocess data
        X_train = self.preprocess_data(data, self.attribute_type)
        self.data_shape = X_train.shape[1]
        
        # Preprocess conditions
        y_train = self.preprocess_conditions(conditions)
        self.condition_dim = y_train.shape[1]
        
        # Build models
        self.generator = self.build_generator(self.data_shape)
        self.discriminator = self.build_discriminator(self.data_shape)
        self.cgan = self.build_cgan(self.generator, self.discriminator)
        
        # Labels for real and fake data
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of data
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            real_conditions = y_train[idx]
            
            # Generate a batch of new data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_data = self.generator.predict([noise, real_conditions])
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([real_data, real_conditions], real)
            d_loss_fake = self.discriminator.train_on_batch([gen_data, real_conditions], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Generate a batch of noise and conditions
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_conditions = y_train[np.random.randint(0, y_train.shape[0], batch_size)]
            
            # Train the generator
            g_loss = self.cgan.train_on_batch([noise, sampled_conditions], real)
            
            # Print progress
            if epoch % sample_interval == 0:
                print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
    
    def generate(self, conditions, n_samples=1):
        """
        Generate synthetic data based on conditions.
        
        Parameters:
        -----------
        conditions : list or numpy.ndarray
            Conditions to generate data for
        n_samples : int, default=1
            Number of samples to generate per condition
            
        Returns:
        --------
        pandas.DataFrame
            Generated data
        """
        if self.generator is None:
            raise ValueError("Model has not been trained yet.")
        
        # Preprocess conditions
        processed_conditions = self.preprocess_conditions(conditions)
        
        # Repeat each condition n_samples times
        repeated_conditions = np.repeat(processed_conditions, n_samples, axis=0)
        
        # Generate noise
        noise = np.random.normal(0, 1, (len(repeated_conditions), self.latent_dim))
        
        # Generate data
        gen_data = self.generator.predict([noise, repeated_conditions])
        
        # Convert generated data back to original format
        return self.postprocess_data(gen_data, conditions, n_samples)
    
    def postprocess_data(self, gen_data, conditions, n_samples):
        """
        Convert generated data back to original format.
        
        Parameters:
        -----------
        gen_data : numpy.ndarray
            Generated data
        conditions : list or numpy.ndarray
            Conditions used to generate data
        n_samples : int
            Number of samples generated per condition
            
        Returns:
        --------
        pandas.Series
            Postprocessed data
        """
        # Inverse transform numerical features if needed
        numerical_cols = gen_data.shape[1]
        if hasattr(self.scaler, 'inverse_transform') and numerical_cols > 0:
            gen_data = self.scaler.inverse_transform(gen_data)
        
        # Convert to appropriate format based on attribute type
        if self.attribute_type == 'name':
            # Generate random names based on features
            import random
            import string
            
            first_names = ['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'Robert', 'Lisa', 'William', 'Mary']
            last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor']
            middle_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            
            synthetic_names = []
            for i in range(gen_data.shape[0]):
                name_length = int(gen_data[i, 0] * 20) + 5  # Scale to reasonable name length
                word_count = min(int(gen_data[i, 1] * 3) + 1, 3)  # 1-3 words
                has_middle_name = gen_data[i, 2] > 0.5
                
                first = random.choice(first_names)
                last = random.choice(last_names)
                
                if word_count == 1:
                    name = first
                elif word_count == 2:
                    name = f"{first} {last}"
                else:
                    middle = random.choice(middle_names)
                    name = f"{first} {middle} {last}"
                
                synthetic_names.append(name)
            
            return pd.Series(synthetic_names)
            
        elif self.attribute_type == 'ssn':
            # Generate SSNs
            synthetic_ssns = []
            for i in range(gen_data.shape[0]):
                area = str(int(gen_data[i, 0])).zfill(3)
                group = str(int(gen_data[i, 1])).zfill(2)
                serial = str(int(gen_data[i, 2])).zfill(4)
                ssn = f"{area}-{group}-{serial}"
                synthetic_ssns.append(ssn)
            
            return pd.Series(synthetic_ssns)
            
        elif self.attribute_type == 'email':
            # Generate emails
            import random
            import string
            
            domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 
                      'icloud.com', 'protonmail.com', 'mail.com', 'zoho.com', 'yandex.com']
            
            synthetic_emails = []
            for i in range(gen_data.shape[0]):
                # Last column is username length
                username_length = max(int(gen_data[i, -1] * 15) + 5, 3)  # At least 3 chars
                
                # Generate random username
                username = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(username_length))
                
                # Select domain based on one-hot encoded columns
                domain_probs = gen_data[i, :-1]
                if np.sum(domain_probs) > 0:
                    domain_idx = np.argmax(domain_probs)
                    domain = domains[min(domain_idx, len(domains)-1)]
                else:
                    domain = random.choice(domains)
                
                email = f"{username}@{domain}"
                synthetic_emails.append(email)
            
            return pd.Series(synthetic_emails)
            
        elif self.attribute_type == 'phone':
            # Generate phone numbers
            synthetic_phones = []
            for i in range(gen_data.shape[0]):
                area_code = str(int(abs(gen_data[i, 0]) * 900 + 100)).zfill(3)  # Area codes from 100-999
                prefix = str(random.randint(100, 999))
                line = str(random.randint(1000, 9999))
                phone = f"({area_code}) {prefix}-{line}"
                synthetic_phones.append(phone)
            
            return pd.Series(synthetic_phones)
            
        elif self.attribute_type == 'address':
            # Generate addresses
            import random
            
            street_names = ['Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'Elm', 'Washington', 'Park', 'Lake', 'Hill']
            street_types = ['St', 'Ave', 'Blvd', 'Rd', 'Ln', 'Dr', 'Way', 'Pl', 'Ct', 'Terrace']
            cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
            states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA']
            
            synthetic_addresses = []
            for i in range(gen_data.shape[0]):
                has_street_number = gen_data[i, 1] > 0.5
                has_apt_number = gen_data[i, 2] > 0.5
                
                street_number = str(random.randint(1, 9999)) if has_street_number else ''
                street_name = random.choice(street_names)
                street_type = random.choice(street_types)
                city_idx = random.randint(0, len(cities)-1)
                city = cities[city_idx]
                state = states[city_idx]
                zip_code = str(random.randint(10000, 99999))
                
                apt_part = f" Apt {random.randint(1, 999)}" if has_apt_number else ''
                
                if street_number:
                    address = f"{street_number} {street_name} {street_type}{apt_part}, {city}, {state} {zip_code}"
                else:
                    address = f"{street_name} {street_type}{apt_part}, {city}, {state} {zip_code}"
                
                synthetic_addresses.append(address)
            
            return pd.Series(synthetic_addresses)
            
        elif self.attribute_type == 'birthdate':
            # Generate birthdates
            import datetime
            
            synthetic_birthdates = []
            for i in range(gen_data.shape[0]):
                year = int(gen_data[i, 0] * 100 + 1920)  # Years from 1920 to 2020
                month = max(1, min(12, int(gen_data[i, 1] * 12) + 1))  # Months from 1 to 12
                day = max(1, min(28, int(gen_data[i, 2] * 28) + 1))  # Days from 1 to 28 (to avoid invalid dates)
                
                try:
                    date = datetime.date(year, month, day)
                    birthdate = date.strftime('%Y-%m-%d')
                except ValueError:
                    # Fallback for invalid dates
                    birthdate = f"{year}-{month:02d}-{day:02d}"
                
                synthetic_birthdates.append(birthdate)
            
            return pd.Series(synthetic_birthdates)
            
        elif self.attribute_type == 'credit_card':
            # Generate credit card numbers
            synthetic_cards = []
            for i in range(gen_data.shape[0]):
                bin_iin = str(int(abs(gen_data[i, 0]) * 900000 + 100000)).zfill(6)  # BIN/IIN from 100000-999999
                rest = ''.join([str(random.randint(0, 9)) for _ in range(10)])
                card = f"{bin_iin}{rest}"
                formatted_card = f"{card[:4]}-{card[4:8]}-{card[8:12]}-{card[12:16]}"
                synthetic_cards.append(formatted_card)
            
            return pd.Series(synthetic_cards)
            
        else:
            # For unrecognized types, return as is
            return pd.Series(gen_data.flatten())


def create_cgan_model_for_attribute(attribute_type):
    """
    Create a CGAN model for a specific attribute type.
    
    Parameters:
    -----------
    attribute_type : str
        Type of the attribute ('name', 'ssn', 'email', etc.)
        
    Returns:
    --------
    CGAN
        CGAN model for the attribute type
    """
    return CGAN(latent_dim=100, condition_dim=None)


def train_cgan_model(data, conditions, attribute_type, epochs=2000):
    """
    Train a CGAN model for a specific attribute type.
    
    Parameters:
    -----------
    data : pandas.Series
        Data to train on
    conditions : pandas.Series or numpy.ndarray
        Conditions for the data
    attribute_type : str
        Type of the attribute ('name', 'ssn', 'email', etc.)
    epochs : int, default=2000
        Number of epochs to train for
        
    Returns:
    --------
    CGAN
        Trained CGAN model
    """
    model = create_cgan_model_for_attribute(attribute_type)
    model.attribute_type = attribute_type
    model.fit(data, conditions, epochs=epochs)
    return model


def generate_synthetic_data_with_cgan(model, conditions, n_samples=1):
    """
    Generate synthetic data using a trained CGAN model.
    
    Parameters:
    -----------
    model : CGAN
        Trained CGAN model
    conditions : list or numpy.ndarray
        Conditions to generate data for
    n_samples : int, default=1
        Number of samples to generate per condition
        
    Returns:
    --------
    pandas.Series
        Generated synthetic data
    """
    return model.generate(conditions, n_samples)


def main():
    """
    Main function to demonstrate the CGAN model.
    """
    # Example usage
    from faker import Faker
    import random
    
    fake = Faker()
    
    # Generate sample data
    n_samples = 1000
    names = pd.Series([fake.name() for _ in range(n_samples)])
    
    # Generate conditions (e.g., gender)
    genders = pd.Series([random.choice(['Male', 'Female']) for _ in range(n_samples)])
    
    # Train CGAN model
    print("Training CGAN model...")
    model = train_cgan_model(names, genders, 'name', epochs=200)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    conditions = ['Male', 'Female', 'Male', 'Female', 'Male']
    synthetic_data = generate_synthetic_data_with_cgan(model, conditions, n_samples=2)
    
    print("\nGenerated synthetic data:")
    for i, data in enumerate(synthetic_data):
        print(f"{conditions[i//2]}: {data}")


if __name__ == "__main__":
    main()