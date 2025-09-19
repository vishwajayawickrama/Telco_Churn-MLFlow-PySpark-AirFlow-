import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions

logger = get_logger(__name__)


class DataIngestor(ABC):
    """ Abstract base class for data ingestion strategies. """
    
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a given source.
        
        Args:
            file_path_or_link (str): Path to the data file or URL
            
        Returns:
            pd.DataFrame: The ingested data as a pandas DataFrame
        """
        pass


class DataIngestorCSV(DataIngestor):
    """ CSV file data ingestor implementation. """
    
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """
        Ingest data from a CSV file.
        
        Args:
            file_path_or_link (str): Path to the CSV file or URL
            
        Returns:
            pd.DataFrame: The ingested data as a pandas DataFrame

        """
        ProjectLogger.log_step_header(logger, "STEP", f"STARTING CSV INGESTION FROM: {file_path_or_link}")
        
        try:
            # Read the CSV file
            logger.debug("Reading CSV file")
            df = pd.read_csv(file_path_or_link)
            
            # Validate the loaded data
            if df.empty:
                logger.warning(f"Loaded DataFrame is empty from: {file_path_or_link}")
                raise pd.errors.EmptyDataError("The CSV file contains no data")
            
            # Log successful ingestion details
            ProjectLogger.log_success_header(logger, "CSV INGESTION COMPLETED SUCCESSFULLY")
            logger.info(f"Data Summary:")
            logger.info(f"  - Shape: {df.shape}")
            logger.info(f"  - Columns: {list(df.columns)}")
            logger.info(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Log data quality summary
            logger.info(f"Data Quality Summary:")
            logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
            logger.info(f"  - Duplicate rows: {df.duplicated().sum()}")
            logger.info(f"  - Data types: {df.dtypes.value_counts().to_dict()}")
            ProjectLogger.log_success_header(logger, "CSV INGESTION SUMMARY COMPLETE")
            
            return df
            
        except FileNotFoundError as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - FILE NOT FOUND")
            logger.error(f"File not found error: {str(e)}")
            raise
        except pd.errors.EmptyDataError as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - EMPTY DATA")
            logger.error(f"Empty data error: {str(e)}")
            raise
        except pd.errors.ParserError as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - PARSING ERROR")
            logger.error(f"CSV parsing error: {str(e)}")
            raise
        except PermissionError as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - PERMISSION DENIED")
            logger.error(f"Permission denied accessing file: {str(e)}")
            raise
        except UnicodeDecodeError as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - ENCODING ERROR")
            logger.error(f"Encoding error while reading CSV: {str(e)}")
            logger.info("Try specifying encoding parameter (e.g., encoding='utf-8', 'latin-1', etc.)")
            raise
        except Exception as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - UNEXPECTED ERROR")
            logger.error(f"Unexpected error during CSV ingestion: {str(e)}")
            logger.error("CSV ingestion failed", exc_info=True)
            raise
    
class DataIngestorExcel(DataIngestor):
    """ Excel file data ingestor implementation. """
    
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        """
        Ingest data from an Excel file.
        
        Args:
            file_path_or_link (str): Path to the Excel file or URL
            
        Returns:
            pd.DataFrame: The ingested data as a pandas DataFrame
            
        """
        ProjectLogger.log_step_header(logger, "STEP", f"STARTING EXCEL INGESTION FROM: {file_path_or_link}")
        
        try:
            # Check file extension for local files
            if not file_path_or_link.startswith(('http://', 'https://', 'ftp://')):
                file_ext = os.path.splitext(file_path_or_link)[1].lower()
                if file_ext not in ['.xlsx', '.xls']:
                    logger.warning(f"Unexpected file extension: {file_ext}")
            
            # Read the Excel file
            logger.debug("Reading Excel file")
            df = pd.read_excel(file_path_or_link)
            
            # Validate the loaded data
            if df.empty:
                logger.warning(f"Loaded DataFrame is empty from: {file_path_or_link}")
                raise ValueError("The Excel file contains no data")
            
            # Log successful ingestion details
            ProjectLogger.log_success_header(logger, "EXCEL INGESTION COMPLETED SUCCESSFULLY")
            logger.info(f"Data Summary:")
            logger.info(f"  - Shape: {df.shape}")
            logger.info(f"  - Columns: {list(df.columns)}")
            logger.info(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Log data quality summary
            logger.info(f"Data Quality Summary:")
            logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
            logger.info(f"  - Duplicate rows: {df.duplicated().sum()}")
            logger.info(f"  - Data types: {df.dtypes.value_counts().to_dict()}")
            ProjectLogger.log_success_header(logger, "EXCEL INGESTION SUMMARY COMPLETE")
            
            return df
            
        except FileNotFoundError as e:
            ProjectLogger.log_error_header(logger, "EXCEL INGESTION FAILED - FILE NOT FOUND")
            logger.error(f"File not found error: {str(e)}")
            raise
        except ValueError as e:
            ProjectLogger.log_error_header(logger, "EXCEL INGESTION FAILED - VALUE ERROR")
            logger.error(f"Value error during Excel ingestion: {str(e)}")
            raise
        except PermissionError as e:
            ProjectLogger.log_error_header(logger, "EXCEL INGESTION FAILED - PERMISSION DENIED")
            logger.error(f"Permission denied accessing file: {str(e)}")
            raise
        except ImportError as e:
            ProjectLogger.log_error_header(logger, "EXCEL INGESTION FAILED - MISSING DEPENDENCY")
            logger.error(f"Missing dependency for Excel reading: {str(e)}")
            logger.info("Please install openpyxl or xlrd: pip install openpyxl xlrd")
            raise
        except Exception as e:
            ProjectLogger.log_error_header(logger, "EXCEL INGESTION FAILED - UNEXPECTED ERROR")
            logger.error(f"Unexpected error during Excel ingestion: {str(e)}")
            logger.error("Excel ingestion failed", exc_info=True)
            raise