import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class OutlierDetectionStrategy(ABC):
    """
    Abstract base class for outlier detection strategies.
    """

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers in specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean DataFrame indicating outliers
        """
        pass


class IQROutlierDetection(OutlierDetectionStrategy):
    """
    Outlier detection using Interquartile Range (IQR) method.
    """
    
    def __init__(self, multiplier: float = 1.5):
        """
        Initialize IQR outlier detection.
        
        Args:
            multiplier (float): IQR multiplier for outlier threshold (default: 1.5)
        """
        self.multiplier = multiplier
        
        ProjectLogger.log_section_header(logger, "INITIALIZING IQR OUTLIER DETECTION")
        logger.info(f"IQR multiplier: {self.multiplier}")
        logger.info("Method: Values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR are considered outliers")

    @log_exceptions(logger)
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean DataFrame indicating outliers
            
        Raises:
            ValueError: If DataFrame is empty or columns don't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "DETECTING OUTLIERS USING IQR METHOD")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if not columns:
                logger.warning("No columns specified for outlier detection")
                return pd.DataFrame(False, index=df.index, columns=[])
            
            # Check if columns exist
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Columns not found in DataFrame: {missing_columns}")
                existing_columns = [col for col in columns if col in df.columns]
                logger.info(f"Using existing columns: {existing_columns}")
            else:
                existing_columns = columns
            
            if not existing_columns:
                logger.info("No valid columns for outlier detection")
                return pd.DataFrame(False, index=df.index, columns=[])
            
            # Initialize outliers DataFrame
            outliers = pd.DataFrame(False, index=df.index, columns=existing_columns)
            
            logger.info(f"Detecting outliers in {len(existing_columns)} columns:")
            
            total_outliers = 0
            outlier_summary = {}
            
            for col in existing_columns:
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Column '{col}' is not numeric, skipping outlier detection")
                    continue
                
                # Calculate quartiles and IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Calculate outlier bounds
                lower_bound = Q1 - self.multiplier * IQR
                upper_bound = Q3 + self.multiplier * IQR
                
                # Detect outliers
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = col_outliers
                
                # Count outliers in this column
                col_outlier_count = col_outliers.sum()
                total_outliers += col_outlier_count
                outlier_summary[col] = {
                    'count': col_outlier_count,
                    'percentage': (col_outlier_count / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR
                }
                
                logger.info(f"  - {col}: {col_outlier_count} outliers ({(col_outlier_count/len(df))*100:.2f}%)")
                logger.debug(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Log summary
            logger.info(f"Total outliers detected: {total_outliers}")
            logger.info(f"Outlier detection completed for {len(existing_columns)} columns")
            
            # Store summary for potential use
            outliers.outlier_summary = outlier_summary
            
            ProjectLogger.log_success_header(logger, "OUTLIER DETECTION COMPLETED")
            
            return outliers
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN OUTLIER DETECTION")
            logger.error(f"Unexpected error: {str(e)}")
            raise


class OutlierDetector:
    """
    Main class for handling outlier detection and treatment.
    """
    
    def __init__(self, strategy: OutlierDetectionStrategy):
        """
        Initialize outlier detector with a specific strategy.
        
        Args:
            strategy (OutlierDetectionStrategy): Outlier detection strategy to use
        """
        self.strategy = strategy
        
        ProjectLogger.log_section_header(logger, "INITIALIZING OUTLIER DETECTOR")
        logger.info(f"Using strategy: {strategy.__class__.__name__}")

    @log_exceptions(logger)
    def detect_outliers(self, df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers using the configured strategy.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            selected_columns (List[str]): Columns to check for outliers
            
        Returns:
            pd.DataFrame: Boolean DataFrame indicating outliers
        """
        return self.strategy.detect_outliers(df, selected_columns)
    
    @log_exceptions(logger)
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        selected_columns: List[str], 
        method: str = 'remove',
        min_outlier_count: int = 2
    ) -> pd.DataFrame:
        """
        Handle outliers in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            selected_columns (List[str]): Columns to check for outliers
            method (str): Method to handle outliers ('remove', 'cap', 'transform')
            min_outlier_count (int): Minimum number of outlier columns to trigger removal
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
            
        Raises:
            ValueError: If DataFrame is empty or method is invalid
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "HANDLING OUTLIERS")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if method not in ['remove', 'cap', 'transform']:
                raise ValueError(f"Invalid method '{method}'. Must be 'remove', 'cap', or 'transform'")
            
            logger.info(f"Initial DataFrame shape: {df.shape}")
            logger.info(f"Outlier handling method: {method}")
            logger.info(f"Minimum outlier count for removal: {min_outlier_count}")
            
            # Detect outliers
            outliers = self.detect_outliers(df, selected_columns)
            
            if outliers.empty:
                logger.info("No outliers detected, returning original DataFrame")
                return df.copy()
            
            if method == 'remove':
                # Count outliers per row
                outlier_count = outliers.sum(axis=1)
                rows_to_remove = outlier_count >= min_outlier_count
                
                rows_removed = rows_to_remove.sum()
                logger.info(f"Rows with >= {min_outlier_count} outliers: {rows_removed}")
                
                # Remove rows with multiple outliers
                df_cleaned = df[~rows_to_remove].copy()
                
                logger.info(f"Rows removed: {rows_removed}")
                logger.info(f"Final DataFrame shape: {df_cleaned.shape}")
                logger.info(f"Data retention rate: {(len(df_cleaned)/len(df))*100:.2f}%")
                
            elif method == 'cap':
                # Cap outliers to bounds
                logger.info("Capping outliers to bounds")
                df_cleaned = df.copy()
                
                if hasattr(outliers, 'outlier_summary'):
                    for col, summary in outliers.outlier_summary.items():
                        if col in df_cleaned.columns:
                            # Cap values to bounds
                            df_cleaned[col] = df_cleaned[col].clip(
                                lower=summary['lower_bound'],
                                upper=summary['upper_bound']
                            )
                            logger.info(f"  - Capped {col} to [{summary['lower_bound']:.2f}, {summary['upper_bound']:.2f}]")
                
            elif method == 'transform':
                # Log transform (placeholder - can be extended)
                logger.info("Applying log transformation to outliers")
                df_cleaned = df.copy()
                # Implementation can be added based on specific requirements
                
            ProjectLogger.log_success_header(logger, "OUTLIER HANDLING COMPLETED")
            
            return df_cleaned
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN OUTLIER HANDLING")
            logger.error(f"Unexpected error: {str(e)}")
            raise

