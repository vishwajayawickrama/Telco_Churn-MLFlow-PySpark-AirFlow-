import os
import sys
import pandas as pd
from enum import Enum
from typing import Optional
from abc import ABC, abstractmethod

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class MissingValueHandlingStrategy(ABC):
    """
    Abstract base class for missing value handling strategies.
    """
    
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with potential missing values
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to handle missing values by dropping rows with missing values in critical columns.
    """
    
    def __init__(self, critical_columns: list = None):
        """
        Initialize the drop missing values strategy.
        
        Args:
            critical_columns (list): List of column names that are critical for analysis
        """
        self.critical_columns = critical_columns or []
        
        ProjectLogger.log_section_header(logger, "INITIALIZING DROP MISSING VALUES STRATEGY")
        logger.info(f"Critical columns for missing value check: {self.critical_columns}")
        logger.info("Strategy: Drop rows with missing values in critical columns")

    @log_exceptions(logger)
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing values in critical columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with rows containing missing critical values removed
            
        Raises:
            ValueError: If DataFrame is empty or critical columns don't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "DROPPING MISSING VALUES")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            # Log initial state
            initial_rows = len(df)
            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            # Check if critical columns exist
            missing_columns = [col for col in self.critical_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Critical columns not found in DataFrame: {missing_columns}")
                # Use only existing columns
                existing_critical_cols = [col for col in self.critical_columns if col in df.columns]
                logger.info(f"Using existing critical columns: {existing_critical_cols}")
            else:
                existing_critical_cols = self.critical_columns
            
            # Check for missing values in critical columns
            if existing_critical_cols:
                missing_counts = df[existing_critical_cols].isnull().sum()
                total_missing = missing_counts.sum()
                
                if total_missing > 0:
                    logger.info(f"Missing values found in critical columns:")
                    for col, count in missing_counts[missing_counts > 0].items():
                        logger.info(f"  - {col}: {count} missing values")
                    
                    # Drop rows with missing values in critical columns
                    df_cleaned = df.dropna(subset=existing_critical_cols)
                else:
                    logger.info("No missing values found in critical columns")
                    df_cleaned = df.copy()
            else:
                logger.info("No critical columns specified, returning original DataFrame")
                df_cleaned = df.copy()
            
            # Log results
            final_rows = len(df_cleaned)
            n_dropped = initial_rows - final_rows
            
            logger.info(f"Rows dropped: {n_dropped}")
            logger.info(f"Final DataFrame shape: {df_cleaned.shape}")
            logger.info(f"Data retention rate: {(final_rows/initial_rows)*100:.2f}%")
            
            ProjectLogger.log_success_header(logger, "MISSING VALUES HANDLING COMPLETED")
            
            return df_cleaned
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MISSING VALUES HANDLING")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """
    Strategy to handle missing values by filling them with specific values or statistical measures.
    """
    
    def __init__(
        self, 
        method: str = 'mean', 
        fill_value=None, 
        relevant_columns: list = None, 
        is_custom_imputer: bool = False, 
        custom_imputer=None
    ):
        """
        Initialize the fill missing values strategy.
        
        Args:
            method (str): Fill method ('mean', 'median', 'mode', 'constant')
            fill_value: Specific value to fill (for 'constant' method)
            relevant_columns (list): Columns to apply the strategy to
            is_custom_imputer (bool): Whether to use a custom imputer
            custom_imputer: Custom imputer object
        """
        self.method = method
        self.fill_value = fill_value
        self.relevant_columns = relevant_columns or []
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer
        
        ProjectLogger.log_section_header(logger, "INITIALIZING FILL MISSING VALUES STRATEGY")
        logger.info(f"Fill method: {self.method}")
        logger.info(f"Fill value: {self.fill_value}")
        logger.info(f"Relevant columns: {self.relevant_columns}")
        logger.info(f"Using custom imputer: {self.is_custom_imputer}")

    @log_exceptions(logger)
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with missing values
            
        Returns:
            pd.DataFrame: DataFrame with missing values filled
            
        Raises:
            ValueError: If DataFrame is empty or method is invalid
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "FILLING MISSING VALUES")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            # Log initial state
            logger.info(f"Initial DataFrame shape: {df.shape}")
            initial_missing = df.isnull().sum().sum()
            logger.info(f"Total missing values: {initial_missing}")
            
            # Use custom imputer if specified
            if self.is_custom_imputer:
                if self.custom_imputer is None:
                    raise ValueError("Custom imputer is None but is_custom_imputer is True")
                
                logger.info("Using custom imputer for missing values")
                df_filled = self.custom_imputer.impute(df)
                
            else:
                # Use built-in filling methods
                df_filled = df.copy()
                
                # Check if relevant columns exist
                if self.relevant_columns:
                    missing_columns = [col for col in self.relevant_columns if col not in df.columns]
                    if missing_columns:
                        logger.warning(f"Relevant columns not found in DataFrame: {missing_columns}")
                        existing_relevant_cols = [col for col in self.relevant_columns if col in df.columns]
                        logger.info(f"Using existing relevant columns: {existing_relevant_cols}")
                    else:
                        existing_relevant_cols = self.relevant_columns
                else:
                    # Use all numeric columns
                    existing_relevant_cols = df.select_dtypes(include=['number']).columns.tolist()
                    logger.info(f"No relevant columns specified, using all numeric columns: {existing_relevant_cols}")
                
                if existing_relevant_cols:
                    # Log missing values in relevant columns
                    missing_in_relevant = df[existing_relevant_cols].isnull().sum()
                    columns_with_missing = missing_in_relevant[missing_in_relevant > 0]
                    
                    if len(columns_with_missing) > 0:
                        logger.info(f"Missing values in relevant columns:")
                        for col, count in columns_with_missing.items():
                            logger.info(f"  - {col}: {count} missing values")
                        
                        # Apply filling strategy
                        if self.method == 'mean':
                            logger.info("Filling missing values with mean")
                            df_filled[existing_relevant_cols] = df_filled[existing_relevant_cols].fillna(
                                df_filled[existing_relevant_cols].mean()
                            )
                        elif self.method == 'median':
                            logger.info("Filling missing values with median")
                            df_filled[existing_relevant_cols] = df_filled[existing_relevant_cols].fillna(
                                df_filled[existing_relevant_cols].median()
                            )
                        elif self.method == 'mode':
                            logger.info("Filling missing values with mode")
                            for col in existing_relevant_cols:
                                if df_filled[col].isnull().any():
                                    mode_val = df_filled[col].mode()
                                    if not mode_val.empty:
                                        df_filled[col].fillna(mode_val[0], inplace=True)
                        elif self.method == 'constant':
                            logger.info(f"Filling missing values with constant: {self.fill_value}")
                            df_filled[existing_relevant_cols] = df_filled[existing_relevant_cols].fillna(self.fill_value)
                        else:
                            raise ValueError(f"Unknown fill method: {self.method}")
                    else:
                        logger.info("No missing values found in relevant columns")
                else:
                    logger.info("No relevant columns to fill")
            
            # Log results
            final_missing = df_filled.isnull().sum().sum()
            filled_count = initial_missing - final_missing
            
            logger.info(f"Missing values filled: {filled_count}")
            logger.info(f"Remaining missing values: {final_missing}")
            logger.info(f"Final DataFrame shape: {df_filled.shape}")
            
            ProjectLogger.log_success_header(logger, "MISSING VALUES FILLING COMPLETED")
            
            return df_filled
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MISSING VALUES FILLING")
            logger.error(f"Unexpected error: {str(e)}")
            raise