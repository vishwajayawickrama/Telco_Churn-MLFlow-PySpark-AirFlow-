import os
import sys
import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Optional, Union
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add utils to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)


class ScalingType(str, Enum):
    """Enumeration for scaling types."""
    MINMAX = 'minmax'
    STANDARD = 'standard'


class FeatureScalingStrategy(ABC):
    """
    Abstract base class for feature scaling strategies.
    """

    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Scale features in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        pass


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """
    Min-Max scaling strategy using sklearn's MinMaxScaler.
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        """
        Initialize Min-Max scaling strategy.
        
        Args:
            feature_range (tuple): Range for scaled features (default: (0, 1))
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.fitted = False
        self.feature_range = feature_range
        self.scaling_stats = {}
        
        ProjectLogger.log_section_header(logger, "INITIALIZING MIN-MAX SCALING STRATEGY")
        logger.info(f"Feature range: {self.feature_range}")
        logger.info("Method: (X - min) / (max - min) * (range_max - range_min) + range_min")

    @log_exceptions(logger)
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Apply Min-Max scaling to specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
            
        Raises:
            ValueError: If DataFrame is empty or columns don't exist
            Exception: For any unexpected errors
        """
        ProjectLogger.log_step_header(logger, "STEP", "APPLYING MIN-MAX SCALING")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if not columns_to_scale:
                logger.warning("No columns specified for scaling")
                return df.copy()
            
            logger.info(f"Scaling {len(columns_to_scale)} columns: {columns_to_scale}")
            logger.info(f"Initial DataFrame shape: {df.shape}")
            
            # Create copy to avoid modifying original
            df_result = df.copy()
            
            # Validate columns exist
            missing_columns = [col for col in columns_to_scale if col not in df_result.columns]
            if missing_columns:
                logger.warning(f"Columns not found: {missing_columns}")
                existing_columns = [col for col in columns_to_scale if col in df_result.columns]
                logger.info(f"Using existing columns: {existing_columns}")
            else:
                existing_columns = columns_to_scale
            
            if not existing_columns:
                logger.warning("No valid columns for scaling")
                return df_result
            
            # Special handling for TotalCharges column (if present)
            if "TotalCharges" in existing_columns:
                logger.info("Converting TotalCharges to numeric (handling potential string values)")
                original_type = df_result["TotalCharges"].dtype
                df_result["TotalCharges"] = pd.to_numeric(df_result["TotalCharges"], errors="coerce")
                
                # Check for conversion issues
                conversion_issues = df_result["TotalCharges"].isnull().sum() - df["TotalCharges"].isnull().sum()
                if conversion_issues > 0:
                    logger.warning(f"TotalCharges conversion created {conversion_issues} NaN values")
                
                logger.info(f"TotalCharges converted: {original_type} -> {df_result['TotalCharges'].dtype}")
            
            # Prepare data for scaling
            scaling_data = df_result[existing_columns].copy()
            
            # Log original statistics
            logger.info("Original data statistics:")
            for col in existing_columns:
                if pd.api.types.is_numeric_dtype(scaling_data[col]):
                    stats = {
                        'min': scaling_data[col].min(),
                        'max': scaling_data[col].max(),
                        'mean': scaling_data[col].mean(),
                        'std': scaling_data[col].std(),
                        'missing': scaling_data[col].isnull().sum()
                    }
                    logger.info(f"  - {col}: min={stats['min']:.3f}, max={stats['max']:.3f}, "
                              f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, missing={stats['missing']}")
                    self.scaling_stats[col] = stats
                else:
                    logger.warning(f"  - {col}: Not numeric, skipping scaling")
                    existing_columns.remove(col)
            
            if not existing_columns:
                logger.warning("No numeric columns available for scaling")
                return df_result
            
            # Handle missing values before scaling
            missing_values = scaling_data[existing_columns].isnull().sum().sum()
            if missing_values > 0:
                logger.warning(f"Found {missing_values} missing values in columns to scale")
                logger.warning("Missing values may affect scaling results")
            
            # Apply scaling
            logger.info("Applying Min-Max scaling transformation...")
            scaled_data = self.scaler.fit_transform(scaling_data[existing_columns])
            self.fitted = True
            
            # Update DataFrame with scaled values
            df_result[existing_columns] = scaled_data
            
            # Log scaled statistics
            logger.info("Scaled data statistics:")
            for i, col in enumerate(existing_columns):
                scaled_col = scaled_data[:, i]
                logger.info(f"  - {col}: min={scaled_col.min():.3f}, max={scaled_col.max():.3f}, "
                          f"mean={scaled_col.mean():.3f}, std={scaled_col.std():.3f}")
            
            # Verify scaling range
            logger.info(f"Scaling verification (should be in range {self.feature_range}):")
            for col in existing_columns:
                col_min = df_result[col].min()
                col_max = df_result[col].max()
                in_range = (self.feature_range[0] <= col_min <= self.feature_range[1] and 
                           self.feature_range[0] <= col_max <= self.feature_range[1])
                logger.info(f"  - {col}: [{col_min:.3f}, {col_max:.3f}] - {'✓' if in_range else '✗'}")
            
            logger.info(f"Min-Max scaling completed for {len(existing_columns)} columns")
            logger.info(f"Final DataFrame shape: {df_result.shape}")
            
            ProjectLogger.log_success_header(logger, "MIN-MAX SCALING COMPLETED")
            
            return df_result
            
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
            logger.error(f"Data validation error: {str(e)}")
            raise
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN MIN-MAX SCALING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def get_scaler(self):
        """
        Get the fitted scaler object.
        
        Returns:
            MinMaxScaler: The fitted scaler
            
        Raises:
            RuntimeError: If scaler is not fitted yet
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call scale() first.")
        return self.scaler
    
    def get_scaling_stats(self) -> dict:
        """
        Get scaling statistics for all processed columns.
        
        Returns:
            dict: Dictionary containing scaling statistics
        """
        return self.scaling_stats


class StandardScalingStrategy(FeatureScalingStrategy):
    """
    Standard scaling strategy using sklearn's StandardScaler.
    """
    
    def __init__(self):
        """Initialize Standard scaling strategy."""
        self.scaler = StandardScaler()
        self.fitted = False
        self.scaling_stats = {}
        
        ProjectLogger.log_section_header(logger, "INITIALIZING STANDARD SCALING STRATEGY")
        logger.info("Method: (X - mean) / std")

    @log_exceptions(logger)
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Apply Standard scaling to specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        ProjectLogger.log_step_header(logger, "STEP", "APPLYING STANDARD SCALING")
        
        try:
            # Validate input
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if not columns_to_scale:
                logger.warning("No columns specified for scaling")
                return df.copy()
            
            logger.info(f"Scaling {len(columns_to_scale)} columns: {columns_to_scale}")
            
            # Create copy and apply scaling logic similar to MinMaxScaling
            df_result = df.copy()
            
            # Validate columns exist
            existing_columns = [col for col in columns_to_scale if col in df_result.columns]
            
            if not existing_columns:
                logger.warning("No valid columns for scaling")
                return df_result
            
            # Apply scaling
            scaled_data = self.scaler.fit_transform(df_result[existing_columns])
            self.fitted = True
            
            # Update DataFrame
            df_result[existing_columns] = scaled_data
            
            logger.info(f"Standard scaling completed for {len(existing_columns)} columns")
            
            ProjectLogger.log_success_header(logger, "STANDARD SCALING COMPLETED")
            
            return df_result
            
        except Exception as e:
            ProjectLogger.log_error_header(logger, "UNEXPECTED ERROR IN STANDARD SCALING")
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def get_scaler(self):
        """Get the fitted scaler object."""
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call scale() first.")
        return self.scaler


class FeatureScaler:
    """
    Main class for handling feature scaling operations.
    """
    
    def __init__(self, strategy: FeatureScalingStrategy):
        """
        Initialize feature scaler with a specific strategy.
        
        Args:
            strategy (FeatureScalingStrategy): Scaling strategy to use
        """
        self.strategy = strategy
        
        ProjectLogger.log_section_header(logger, "INITIALIZING FEATURE SCALER")
        logger.info(f"Using strategy: {strategy.__class__.__name__}")

    @log_exceptions(logger)
    def scale_features(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        """
        Scale features using the configured strategy.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns_to_scale (List[str]): Columns to scale
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        return self.strategy.scale(df, columns_to_scale)