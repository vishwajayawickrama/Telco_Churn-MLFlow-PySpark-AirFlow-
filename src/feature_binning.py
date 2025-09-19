import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class FeatureBinningStrategy(ABC):
    """
    Abstract base class for feature binning strategies.
    """

    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Bin a feature in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to bin
            
        Returns:
            pd.DataFrame: DataFrame with binned feature
        """
        pass


class CustomBinningStrategy(FeatureBinningStrategy):
    """
    Custom feature binning strategy using user-defined bin definitions.
    """
    
    def __init__(self, bin_definitions: Dict[str, List[Union[int, float]]]):
        """
        Initialize custom binning strategy.
        
        Args:
            bin_definitions (Dict): Dictionary defining bins with format:
                {
                    'bin_label': [min_value, max_value],  # Range bin
                    'bin_label': [min_value],             # Open-ended bin (value >= min_value)
                }
        """
        self.bin_definitions = bin_definitions
        
        ProjectLogger.log_section_header(logger, "INITIALIZING CUSTOM BINNING STRATEGY")
        logger.info(f"Number of bin definitions: {len(self.bin_definitions)}")
        
        for bin_label, bin_range in self.bin_definitions.items():
            if len(bin_range) == 2:
                logger.info(f"  - {bin_label}: [{bin_range[0]}, {bin_range[1]}]")
            elif len(bin_range) == 1:
                logger.info(f"  - {bin_label}: >= {bin_range[0]}")
            else:
                logger.warning(f"  - {bin_label}: Invalid range definition {bin_range}")

    @log_exceptions(logger)
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply custom binning to a feature.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to bin
            
        Returns:
            pd.DataFrame: DataFrame with binned feature
            
        Raises:
            ValueError: If DataFrame is empty or column doesn't exist
            KeyError: If column is not found in DataFrame
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", f"APPLYING CUSTOM BINNING TO COLUMN: {column}")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            
            # Check if column is numeric
            if not pd.api.types.is_numeric_dtype(df[column]):
                logger.warning(f"Column '{column}' is not numeric, proceeding with binning anyway")
            
            logger.info(f"Original column shape: {df[column].shape}")
            logger.info(f"Original column type: {df[column].dtype}")
            logger.info(f"Value range: [{df[column].min():.2f}, {df[column].max():.2f}]")
            
            # Create copy to avoid modifying original
            df_result = df.copy()
            
            # Create binned column name
            binned_column = f'{column}Bins'
            
            # Apply binning function
            def assign_bin(value):
                """
                Assign a bin label to a value based on bin definitions.
                
                Args:
                    value: The value to bin
                    
                Returns:
                    str: Bin label
                """
                # Handle missing values
                if pd.isna(value):
                    return "Missing"
                
                # Check each bin definition
                for bin_label, bin_range in self.bin_definitions.items():
                    if len(bin_range) == 2:
                        # Range bin: [min, max]
                        if bin_range[0] <= value <= bin_range[1]:
                            return bin_label
                    elif len(bin_range) == 1:
                        # Open-ended bin: >= min
                        if value >= bin_range[0]:
                            return bin_label
                
                # If no bin matches, return Invalid
                return "Invalid"
            
            # Apply binning
            logger.info("Applying binning transformation...")
            df_result[binned_column] = df_result[column].apply(assign_bin)
            
            # Log binning results
            bin_counts = df_result[binned_column].value_counts()
            logger.info(f"Binning results for '{binned_column}':")
            for bin_label, count in bin_counts.items():
                percentage = (count / len(df_result)) * 100
                logger.info(f"  - {bin_label}: {count} records ({percentage:.2f}%)")
            
            # Check for invalid bins
            invalid_count = (df_result[binned_column] == "Invalid").sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} values that didn't match any bin definition")
                invalid_values = df_result[df_result[binned_column] == "Invalid"][column].unique()
                logger.warning(f"Invalid values sample: {invalid_values[:10]}")  # Show first 10
            
            # Remove original column
            logger.info(f"Removing original column '{column}'")
            del df_result[column]
            
            logger.info(f"Final DataFrame shape: {df_result.shape}")
            logger.info(f"New binned column '{binned_column}' created successfully")
            
            ProjectLogger.log_success_header(logger, "CUSTOM BINNING COMPLETED")
            
            return df_result
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except KeyError as e:
            ProjectLogger.log_error_header(logger, "COLUMN NOT FOUND ERROR")
            logger.error(f"Column error: {str(e)}")
            logger.error(f"Available columns: {list(df.columns)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN FEATURE BINNING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class FeatureBinner:
    """
    Main class for handling feature binning operations.
    """
    
    def __init__(self, strategy: FeatureBinningStrategy):
        """
        Initialize feature binner with a specific strategy.
        
        Args:
            strategy (FeatureBinningStrategy): Binning strategy to use
        """
        self.strategy = strategy
        
        ProjectLogger.log_section_header(logger, "INITIALIZING FEATURE BINNER")
        logger.info(f"Using strategy: {strategy.__class__.__name__}")

    @log_exceptions(logger)
    def apply_binning(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply binning to multiple columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to bin
            
        Returns:
            pd.DataFrame: DataFrame with binned features
            
        Raises:
            ValueError: If DataFrame is empty or no columns specified
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "APPLYING BINNING TO MULTIPLE COLUMNS")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if not columns:
                logger.warning("No columns specified for binning")
                return df.copy()
            
            logger.info(f"Applying binning to {len(columns)} columns: {columns}")
            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            # Apply binning to each column
            df_result = df.copy()
            
            for i, column in enumerate(columns, 1):
                logger.info(f"Processing column {i}/{len(columns)}: {column}")
                
                if column not in df_result.columns:
                    logger.warning(f"Column '{column}' not found, skipping")
                    continue
                
                # Apply binning strategy
                df_result = self.strategy.bin_feature(df_result, column)
                
                logger.info(f"Completed binning for column: {column}")
            
            logger.info(f"Final DataFrame shape: {df_result.shape}")
            logger.info(f"Binning completed for all specified columns")
            
            ProjectLogger.log_success_header(logger, "MULTI-COLUMN BINNING COMPLETED")
            
            return df_result
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MULTI-COLUMN BINNING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
