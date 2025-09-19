"""
Centralized logging configuration for the Telco Customer Churn Prediction project.

This module provides a consistent logging setup across all components of the ML pipeline,
ensuring uniform formatting, proper log levels, and standardized error handling with colored output.
"""

import logging
import os
from typing import Optional
import colorlog


class ProjectLogger:
    """
    Centralized logger configuration for the project.
    
    This class provides consistent logging configuration across all modules
    with standardized formatting and error handling.
    """
    
    _loggers = {}
    
    @classmethod
    def get_logger(
        cls, 
        name: str, 
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Get or create a logger with standardized configuration.
        
        Args:
            name (str): Name of the logger (usually __name__)
            log_file (Optional[str]): Optional log file path
            level (int): Logging level (default: INFO)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Create colored formatter for console
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        
        # Standard formatter for file output
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with colored output
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Cache the logger
        cls._loggers[name] = logger
        
        return logger
    
    @staticmethod
    def log_section_header(logger: logging.Logger, title: str, width: int = 80):
        """
        Log a standardized section header with enhanced formatting.
        
        Args:
            logger (logging.Logger): Logger instance
            title (str): Section title
            width (int): Width of the header line
        """
        logger.info("=" * width)
        logger.info(f"🚀 {title.upper()}")
        logger.info("=" * width)
    
    @staticmethod
    def log_step_header(logger: logging.Logger, step: str, title: str, width: int = 80):
        """
        Log a standardized step header with enhanced formatting.
        
        Args:
            logger (logging.Logger): Logger instance
            step (str): Step number/identifier
            title (str): Step title
            width (int): Width of the header line
        """
        logger.info("\n" + "="*width)
        logger.info(f"⚡ {step}: {title.upper()}")
        logger.info("="*width)
    
    @staticmethod
    def log_error_header(logger: logging.Logger, error_type: str, width: int = 80):
        """
        Log a standardized error header with enhanced formatting.
        
        Args:
            logger (logging.Logger): Logger instance
            error_type (str): Type of error
            width (int): Width of the header line
        """
        logger.error("=" * width)
        logger.error(f"❌ ERROR: {error_type.upper()}")
        logger.error("=" * width)
    
    @staticmethod
    def log_success_header(logger: logging.Logger, message: str, width: int = 80):
        """
        Log a standardized success header with enhanced formatting.
        
        Args:
            logger (logging.Logger): Logger instance
            message (str): Success message
            width (int): Width of the header line
        """
        logger.info("=" * width)
        logger.info(f"✅ SUCCESS: {message.upper()}")
        logger.info("=" * width)


# Convenience function for quick logger setup
def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a standardized logger.
    
    Args:
        name (str): Logger name (usually __name__)
        log_file (Optional[str]): Optional log file path
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return ProjectLogger.get_logger(name, log_file)


# Standard exception handling decorators
def log_exceptions(logger: logging.Logger):
    """
    Decorator to add standardized exception logging to functions.
    
    Args:
        logger (logging.Logger): Logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ProjectLogger.log_error_header(logger, f"{func.__name__} failed")
                logger.error(f"Function: {func.__name__}")
                logger.error(f"Error: {str(e)}")
                logger.error(f"Args: {args}")
                logger.error(f"Kwargs: {kwargs}")
                logger.error("Exception details:", exc_info=True)
                raise
        return wrapper
    return decorator