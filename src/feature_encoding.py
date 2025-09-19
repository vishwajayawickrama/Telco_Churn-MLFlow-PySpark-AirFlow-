import os
import sys
import json
import pandas as pd
from enum import Enum
from typing import Dict, List, Union, Any
from abc import ABC, abstractmethod

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class VariableType(str, Enum):
    """Enumeration for variable types in feature encoding."""
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'


class FeatureEncodingStrategy(ABC):
    """
    Abstract base class for feature encoding strategies.
    """

    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode features in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        pass


class NominalEncodingStrategy(FeatureEncodingStrategy):
    """
    Nominal encoding strategy using label encoding for categorical variables.
    """

    def __init__(self, nominal_columns: List[str]):
        """
        Initialize nominal encoding strategy.
        
        Args:
            nominal_columns (List[str]): List of nominal columns to encode
        """
        self.nominal_columns = nominal_columns
        self.encoder_dict = {}
        
        ProjectLogger.log_section_header(logger, "INITIALIZING NOMINAL ENCODING STRATEGY")
        logger.info(f"Nominal columns to encode: {len(self.nominal_columns)}")
        
        for column in self.nominal_columns:
            logger.info(f"  - {column}")
        
        # Ensure artifacts directory exists
        try:
            os.makedirs('artifacts/encode', exist_ok=True)
            logger.info("Artifacts/encode directory ready")
        except Exception as e:
            logger.warning(f"Could not create artifacts directory: {str(e)}")

    @log_exceptions(logger)
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode nominal variables using label encoding.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with encoded nominal variables
            
        Raises:
            ValueError: If DataFrame is empty or columns don't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "ENCODING NOMINAL VARIABLES")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if not self.nominal_columns:
                logger.warning("No nominal columns specified for encoding")
                return df.copy()
            
            logger.info(f"Starting nominal encoding for {len(self.nominal_columns)} columns")
            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            # Create copy to avoid modifying original
            df_result = df.copy()
            
            # Track encoding results
            encoding_summary = {}
            
            for i, column in enumerate(self.nominal_columns, 1):
                logger.info(f"Processing column {i}/{len(self.nominal_columns)}: {column}")
                
                # Check if column exists
                if column not in df_result.columns:
                    logger.warning(f"Column '{column}' not found in DataFrame, skipping")
                    continue
                
                # Get unique values
                unique_values = df_result[column].unique()
                unique_count = len(unique_values)
                
                logger.info(f"  - Unique values in '{column}': {unique_count}")
                
                # Handle missing values
                has_missing = df_result[column].isnull().any()
                if has_missing:
                    missing_count = df_result[column].isnull().sum()
                    logger.warning(f"  - Missing values in '{column}': {missing_count}")
                
                # Create encoding mapping
                encoder_dict = {str(value): i for i, value in enumerate(unique_values)}
                mapping_dict = {value: i for i, value in enumerate(unique_values)}
                
                # Store encoder for this column
                self.encoder_dict[column] = mapping_dict
                
                # Save encoder to file
                try:
                    encoder_path = os.path.join('artifacts/encode', f"{column}_encoder.json")
                    with open(encoder_path, "w") as f:
                        json.dump(encoder_dict, f, indent=2)
                    logger.info(f"  - Encoder saved to: {encoder_path}")
                except Exception as e:
                    logger.warning(f"  - Could not save encoder for '{column}': {str(e)}")
                
                # Apply encoding
                original_type = df_result[column].dtype
                df_result[column] = df_result[column].map(mapping_dict)
                new_type = df_result[column].dtype
                
                # Check for unmapped values (NaN after mapping)
                unmapped_count = df_result[column].isnull().sum() - (missing_count if has_missing else 0)
                if unmapped_count > 0:
                    logger.warning(f"  - Unmapped values after encoding: {unmapped_count}")
                
                # Store encoding summary
                encoding_summary[column] = {
                    'unique_values': unique_count,
                    'original_type': str(original_type),
                    'new_type': str(new_type),
                    'missing_values': missing_count if has_missing else 0,
                    'unmapped_values': unmapped_count,
                    'encoder_size': len(mapping_dict)
                }
                
                logger.info(f"  - Successfully encoded '{column}': {original_type} -> {new_type}")
            
            # Log encoding summary
            logger.info("Nominal encoding summary:")
            for column, summary in encoding_summary.items():
                logger.info(f"  - {column}: {summary['unique_values']} unique -> {summary['encoder_size']} mappings")
                if summary['unmapped_values'] > 0:
                    logger.warning(f"    WARNING: {summary['unmapped_values']} unmapped values")
            
            logger.info(f"Final DataFrame shape: {df_result.shape}")
            logger.info(f"Encoded {len(encoding_summary)} nominal columns successfully")
            
            ProjectLogger.log_success_header(logger, "NOMINAL ENCODING COMPLETED")
            
            return df_result
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN NOMINAL ENCODING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    """
    Ordinal encoding strategy using custom mappings for ordered categorical variables.
    """

    def __init__(self, ordinal_mappings: Dict[str, Dict[str, int]]):
        """
        Initialize ordinal encoding strategy.
        
        Args:
            ordinal_mappings (Dict): Dictionary mapping column names to their ordinal mappings
                Example: {'education': {'High School': 1, 'Bachelor': 2, 'Master': 3}}
        """
        self.ordinal_mappings = ordinal_mappings
        
        ProjectLogger.log_section_header(logger, "INITIALIZING ORDINAL ENCODING STRATEGY")
        logger.info(f"Ordinal columns to encode: {len(self.ordinal_mappings)}")

    @log_exceptions(logger)
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode ordinal variables using custom mappings.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with encoded ordinal variables

        """
        ProjectLogger.log_step_header(logger, "STEP", "ENCODING ORDINAL VARIABLES")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if not self.ordinal_mappings:
                logger.warning("No ordinal mappings specified for encoding")
                return df.copy()
            
            logger.info(f"Starting ordinal encoding for {len(self.ordinal_mappings)} columns")

            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            for column, mapping in self.ordinal_mappings.items():
                logger.info(f"Processing column: {column}")
        
                df[column] = df[column].map(mapping)
                logger.info(f"  - Successfully encoded '{column}'")

            logger.info(f"Final DataFrame shape: {df.shape}")
            logger.info(f"Encoded {len(self.ordinal_mappings)} ordinal columns successfully")

            ProjectLogger.log_success_header(logger, "ORDINAL ENCODING COMPLETED")

            return df

        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN ORDINAL ENCODING")
            logger.error(f"Unexpected error: {str(e)}")
            raise



