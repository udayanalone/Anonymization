import pandas as pd
import numpy as np
import uuid
import hashlib
import json
from datetime import datetime
from pathlib import Path

class Pseudonymizer:
    """
    Class for pseudonymizing patient identifiers while maintaining referential integrity.
    """
    
    def __init__(self, mapping_file=None):
        """
        Initialize the pseudonymizer.
        
        Parameters:
        -----------
        mapping_file : str, optional
            Path to save/load the pseudonymization mapping
        """
        self.mapping = {}
        self.mapping_file = mapping_file or "pseudonymization_mapping.json"
        self.load_mapping()
    
    def load_mapping(self):
        """Load existing pseudonymization mapping if available."""
        try:
            if Path(self.mapping_file).exists():
                with open(self.mapping_file, 'r') as f:
                    self.mapping = json.load(f)
                print(f"Loaded existing mapping with {len(self.mapping)} entries")
        except Exception as e:
            print(f"Could not load existing mapping: {e}")
            self.mapping = {}
    
    def save_mapping(self):
        """Save the current pseudonymization mapping."""
        try:
            with open(self.mapping_file, 'w') as f:
                json.dump(self.mapping, f, indent=2)
            print(f"Saved mapping to {self.mapping_file}")
        except Exception as e:
            print(f"Error saving mapping: {e}")
    
    def pseudonymize_id(self, original_id):
        """
        Pseudonymize a patient ID using UUID.
        
        Parameters:
        -----------
        original_id : str
            Original patient ID
            
        Returns:
        --------
        str
            Pseudonymized ID
        """
        if original_id in self.mapping:
            return self.mapping[original_id]
        
        # Generate new pseudonymized ID
        pseudonymized_id = str(uuid.uuid4())
        self.mapping[original_id] = pseudonymized_id
        return pseudonymized_id
    
    def pseudonymize_ssn(self, ssn):
        """
        Pseudonymize SSN by hashing.
        
        Parameters:
        -----------
        ssn : str
            Original SSN
            
        Returns:
        --------
        str
            Pseudonymized SSN
        """
        if pd.isna(ssn) or ssn == '':
            return ssn
        
        # Hash the SSN to maintain consistency
        hashed_ssn = hashlib.sha256(str(ssn).encode()).hexdigest()[:9]
        # Format as XXX-XX-XXXX
        return f"{hashed_ssn[:3]}-{hashed_ssn[3:5]}-{hashed_ssn[5:9]}"
    
    def pseudonymize_name(self, first_name, last_name):
        """
        Pseudonymize names by replacing with generic identifiers.
        
        Parameters:
        -----------
        first_name : str
            Original first name
        last_name : str
            Original last name
            
        Returns:
        --------
        tuple
            (pseudonymized_first_name, pseudonymized_last_name)
        """
        if pd.isna(first_name) or first_name == '':
            first_name = "Unknown"
        if pd.isna(last_name) or last_name == '':
            last_name = "Unknown"
        
        # Generate consistent pseudonyms based on original names
        first_hash = hashlib.md5(str(first_name).encode()).hexdigest()[:8]
        last_hash = hashlib.md5(str(last_name).encode()).hexdigest()[:8]
        
        pseudonymized_first = f"Patient_{first_hash}"
        pseudonymized_last = f"ID_{last_hash}"
        
        return pseudonymized_first, pseudonymized_last
    
    def pseudonymize_address(self, address, city, state, zip_code):
        """
        Pseudonymize address by keeping only city and state, masking specific details.
        
        Parameters:
        -----------
        address : str
            Original street address
        city : str
            City name
        state : str
            State name
        zip_code : str
            ZIP code
            
        Returns:
        --------
        tuple
            (pseudonymized_address, pseudonymized_city, pseudonymized_state, pseudonymized_zip)
        """
        # Keep city and state as they are quasi-identifiers
        pseudonymized_city = city
        pseudonymized_state = state
        
        # Mask address details
        if pd.isna(address) or address == '':
            pseudonymized_address = "Address_Redacted"
        else:
            # Keep only the street type (e.g., "Street", "Avenue", "Lane")
            address_parts = str(address).split()
            if len(address_parts) > 1:
                pseudonymized_address = f"Street_{len(address_parts)}_Redacted"
            else:
                pseudonymized_address = "Address_Redacted"
        
        # Mask ZIP code to first 3 digits for privacy
        if pd.isna(zip_code) or zip_code == '':
            pseudonymized_zip = "00000"
        else:
            zip_str = str(zip_code).split('.')[0]  # Remove decimal if present
            if len(zip_str) >= 3:
                pseudonymized_zip = f"{zip_str[:3]}XX"
            else:
                pseudonymized_zip = "00000"
        
        return pseudonymized_address, pseudonymized_city, pseudonymized_state, pseudonymized_zip
    
    def pseudonymize_dataset(self, df):
        """
        Apply pseudonymization to the entire dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Original patient dataset
            
        Returns:
        --------
        pandas.DataFrame
            Pseudonymized dataset
        """
        print("Starting pseudonymization process...")
        
        # Create a copy to avoid modifying original
        df_pseudo = df.copy()
        
        # Pseudonymize identifiers
        print("Pseudonymizing patient IDs...")
        df_pseudo['Id'] = df_pseudo['Id'].apply(self.pseudonymize_id)
        
        print("Pseudonymizing SSNs...")
        df_pseudo['SSN'] = df_pseudo['SSN'].apply(self.pseudonymize_ssn)
        
        print("Pseudonymizing names...")
        df_pseudo[['FIRST', 'LAST']] = df_pseudo.apply(
            lambda row: pd.Series(self.pseudonymize_name(row['FIRST'], row['LAST'])), 
            axis=1
        )
        
        print("Pseudonymizing addresses...")
        df_pseudo[['ADDRESS', 'CITY', 'STATE', 'ZIP']] = df_pseudo.apply(
            lambda row: pd.Series(self.pseudonymize_address(
                row['ADDRESS'], row['CITY'], row['STATE'], row['ZIP']
            )), 
            axis=1
        )
        
        # Remove or mask other sensitive fields
        sensitive_fields = ['DRIVERS', 'PASSPORT', 'MAIDEN', 'LAT', 'LON']
        for field in sensitive_fields:
            if field in df_pseudo.columns:
                print(f"Masking {field}...")
                df_pseudo[field] = f"{field}_Redacted"
        
        # Save the mapping
        self.save_mapping()
        
        print("Pseudonymization completed!")
        return df_pseudo
    
    def get_mapping_stats(self):
        """Get statistics about the pseudonymization mapping."""
        return {
            'total_mappings': len(self.mapping),
            'mapping_file': self.mapping_file
        }

def pseudonymize_patient_data(input_file, output_file, mapping_file=None):
    """
    Convenience function to pseudonymize patient data.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to output CSV file
    mapping_file : str, optional
        Path to save pseudonymization mapping
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Load data
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Initialize pseudonymizer
        pseudonymizer = Pseudonymizer(mapping_file)
        
        # Apply pseudonymization
        df_pseudo = pseudonymizer.pseudonymize_dataset(df)
        
        # Save pseudonymized data
        print(f"Saving pseudonymized data to {output_file}...")
        df_pseudo.to_csv(output_file, index=False)
        
        # Print statistics
        stats = pseudonymizer.get_mapping_stats()
        print(f"Pseudonymization completed successfully!")
        print(f"Total mappings: {stats['total_mappings']}")
        print(f"Mapping saved to: {stats['mapping_file']}")
        
        return True
        
    except Exception as e:
        print(f"Error during pseudonymization: {e}")
        return False
