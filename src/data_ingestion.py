import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from logger import get_logger, ProjectLogger, log_exceptions
from pyspark.sql import DataFrame, SparkSession
from spark_session import get_or_create_spark_session

logger = get_logger(__name__)


class DataIngestor(ABC):
    """ Abstract base class for data ingestion strategies using pyspark. """
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize DataIngestor with a SparkSession.
        
        Args:
            spark: Optional SparkSession. If not provided, will create/get one.
        """
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> DataFrame:
        """
        Abstract method to ingest data from a given source.
        
        Args:
            file_path_or_link (str): Path to the data file or URL
            
        Returns:
            DataFrame: The ingested data as a PySpark DataFrame
        """
        pass


class DataIngestorCSV(DataIngestor):
    """ CSV file data ingestor implementation. """
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        """
        Ingest data from a CSV file.
        
        Args:
            file_path_or_link (str): Path to the CSV file or URL
            **options: Additional options for CSV reading
            
        Returns:
            DataFrame: The ingested data as a PySpark DataFrame

        """
        ProjectLogger.log_step_header(logger, "STEP", f"STARTING CSV INGESTION FROM: {file_path_or_link}")
        
        try:
            # Default CSV options
            csv_options = {
                "header": "true", # First row as header
                "inferSchema": "true", # Infer data types
                "ignoreLeadingWhiteSpace": "true", # Ignore leading whitespace
                "ignoreTrailingWhiteSpace": "true", # Ignore trailing whitespace
                "nullValue": "", # Treat empty strings as null
                "nanValue": "NaN", # Treat 'NaN' as null
                "escape": '"', # Escape character
                "quote": '"' # Quote character
            }
            csv_options.update(options)

            # Read the CSV file
            logger.info("Reading CSV file")
            df = self.spark.read.options(**csv_options).csv(file_path_or_link)
            
            # Get DataFrame info
            row_count = df.count()
            columns = df.columns
            
            # Calculate approximate memory usage
            # Note: This is an estimate as PySpark distributes data
            sample_size = min(1000, row_count)
            if row_count > 0:
                sample_df = df.limit(sample_size).toPandas()
                memory_per_row = sample_df.memory_usage(deep=True).sum() / sample_size
                estimated_memory = (memory_per_row * row_count) / 1024**2
            else:
                estimated_memory = 0
            
            # Log successful ingestion details
            ProjectLogger.log_success_header(logger, "CSV INGESTION COMPLETED SUCCESSFULLY")
            logger.info(f"Data Summary:")
            logger.info(f"Shape: ({row_count}, {len(columns)})")
            logger.info(f"Columns: {columns}")
            logger.info(f"Estimated memory usage: {estimated_memory:.2f} MB")
            logger.info(f"Partitions: {df.rdd.getNumPartitions()}")

            return df
            
        except FileNotFoundError as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - FILE NOT FOUND")
            logger.error(f"File not found error: {str(e)}")
            raise
        except Exception as e:
            ProjectLogger.log_error_header(logger, "CSV INGESTION FAILED - UNEXPECTED ERROR")
            logger.error(f"Unexpected error during CSV ingestion: {str(e)}")
            logger.error("CSV ingestion failed", exc_info=True)
            raise
    
class DataIngestorExcel(DataIngestor):
    """ Excel file data ingestor implementation. """
    
    def ingest(self, file_path_or_link: str, sheet_name: Optional[str] = None, **options) -> DataFrame:
        """
        Ingest Excel data using PySpark.
        Note: This implementation converts Excel to CSV format internally as PySpark
        doesn't have native Excel support. For production use, consider using
        spark-excel library.
        
        Args:
            file_path_or_link (str): Path to the Excel file or URL
            sheet_name: Name of the sheet to read (optional)
            **options: Additional options
            
        Returns:
            DataFrame: The ingested data as a PySpark DataFrame
        """
        ProjectLogger.log_step_header(logger, "STEP", f"STARTING EXCEL INGESTION FROM: {file_path_or_link}")
        
        try:

            
            # Read the Excel file
            logger.debug("Reading Excel file")
            pandas_df = pd.read_excel(file_path_or_link, sheet_name=sheet_name)

            # Convert to PySpark DataFrame
            logger.debug("Converting to PySpark DataFrame")
            df = self.spark.createDataFrame(pandas_df)
            
            # Get DataFrame info
            row_count = df.count()
            columns = df.columns
            
            # Log successful ingestion details
            ProjectLogger.log_success_header(logger, "EXCEL INGESTION COMPLETED SUCCESSFULLY")
            logger.info(f"✓ Shape: ({row_count}, {len(columns)})")
            logger.info(f"Columns: {columns}")
            logger.info(f"Partitions: {df.rdd.getNumPartitions()}")

            return df
            
        except FileNotFoundError as e:
            ProjectLogger.log_error_header(logger, "EXCEL INGESTION FAILED - FILE NOT FOUND")
            logger.error(f"File not found error: {str(e)}")
            raise
        except Exception as e:
            ProjectLogger.log_error_header(logger, "EXCEL INGESTION FAILED - UNEXPECTED ERROR")
            logger.error(f"Unexpected error during Excel ingestion: {str(e)}")
            logger.error("Excel ingestion failed", exc_info=True)
            raise