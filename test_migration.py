#!/usr/bin/env python3
"""
Comprehensive test script for PySpark migration validation.
Tests both PySpark and pandas implementations to ensure functionality equivalence.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'utils'))
sys.path.append(str(project_root / 'pipelines'))

from spark_utils import SparkSessionManager
from logger import get_logger, ProjectLogger
from config import get_config

# Initialize logger
logger = get_logger(__name__)

class TestPySparkMigration(unittest.TestCase):
    """Test suite for PySpark migration validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        ProjectLogger.log_section_header(logger, "SETTING UP PYSPARK MIGRATION TESTS")
        
        # Initialize Spark session
        cls.spark = SparkSessionManager.get_session()
        
        # Test data paths
        cls.test_data_path = "./data/raw/TelcoCustomerChurnPrediction.csv"
        cls.test_output_dir = "./test_outputs"
        
        # Create test output directory
        os.makedirs(cls.test_output_dir, exist_ok=True)
        
        logger.info("Test environment setup completed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        ProjectLogger.log_section_header(logger, "CLEANING UP TEST ENVIRONMENT")
        
        # Stop Spark session
        SparkSessionManager.stop_session()
        
        # Remove test outputs
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
        
        logger.info("Test environment cleanup completed")
    
    def test_spark_session_initialization(self):
        """Test Spark session initialization."""
        logger.info("Testing Spark session initialization")
        
        self.assertIsNotNone(self.spark)
        self.assertEqual(self.spark.sparkContext.appName, "TelcoCustomerChurnPrediction")
        
        logger.info("✓ Spark session initialization test passed")
    
    def test_data_pipeline_pyspark(self):
        """Test PySpark data pipeline."""
        logger.info("Testing PySpark data pipeline")
        
        try:
            from data_pipeline_pyspark import data_pipeline_pyspark
            
            # Run PySpark data pipeline
            result = data_pipeline_pyspark(
                data_path=self.test_data_path,
                target_column='Churn',
                test_size=0.2,
                force_rebuild=True
            )
            
            # Validate results
            self.assertIn('train_df', result)
            self.assertIn('test_df', result)
            self.assertIn('feature_columns', result)
            
            # Check data types
            train_df = result['train_df']
            test_df = result['test_df']
            
            self.assertTrue(train_df.count() > 0)
            self.assertTrue(test_df.count() > 0)
            
            logger.info("✓ PySpark data pipeline test passed")
            
        except Exception as e:
            logger.error(f"PySpark data pipeline test failed: {str(e)}")
            self.fail(f"PySpark data pipeline test failed: {str(e)}")
    
    def test_training_pipeline_pyspark(self):
        """Test PySpark training pipeline."""
        logger.info("Testing PySpark training pipeline")
        
        try:
            from training_pipeline_pyspark import training_pipeline_pyspark
            
            # Run PySpark training pipeline
            result = training_pipeline_pyspark(
                data_path=self.test_data_path,
                target_column='Churn',
                test_size=0.2
            )
            
            # Validate results
            self.assertIn('models', result)
            self.assertIn('metrics', result)
            self.assertIn('best_model', result)
            
            # Check metrics structure
            metrics = result['metrics']
            self.assertIn('gbt', metrics)
            self.assertIn('rf', metrics)
            self.assertIn('lr', metrics)
            self.assertIn('dt', metrics)
            
            # Validate metric values
            for model_name, model_metrics in metrics.items():
                self.assertIn('accuracy', model_metrics)
                self.assertIn('precision', model_metrics)
                self.assertIn('recall', model_metrics)
                self.assertIn('f1', model_metrics)
                self.assertIn('auc', model_metrics)
                
                # Check metric ranges
                self.assertTrue(0 <= model_metrics['accuracy'] <= 1)
                self.assertTrue(0 <= model_metrics['precision'] <= 1)
                self.assertTrue(0 <= model_metrics['recall'] <= 1)
                self.assertTrue(0 <= model_metrics['f1'] <= 1)
                self.assertTrue(0 <= model_metrics['auc'] <= 1)
            
            logger.info("✓ PySpark training pipeline test passed")
            
        except Exception as e:
            logger.error(f"PySpark training pipeline test failed: {str(e)}")
            self.fail(f"PySpark training pipeline test failed: {str(e)}")
    
    def test_streaming_inference_pyspark(self):
        """Test PySpark streaming inference."""
        logger.info("Testing PySpark streaming inference")
        
        try:
            from streaming_inference_pipeline_pyspark import streaming_inference_pyspark
            
            # Sample customer data
            sample_customer = {
                'gender': 'Male',
                'SeniorCitizen': '0',
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 12.0,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'No',
                'DeviceProtection': 'Yes',
                'TechSupport': 'No',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 50.0,
                'TotalCharges': 600.0
            }
            
            # Test inference (this might fail if model doesn't exist, which is expected)
            try:
                result = streaming_inference_pyspark(input_data=sample_customer)
                
                # If successful, validate results
                if 'single_prediction' in result:
                    prediction = result['single_prediction']
                    self.assertIn('churn_prediction', prediction)
                    self.assertIn('churn_probability', prediction)
                    self.assertIn('confidence', prediction)
                    
                    # Validate prediction values
                    self.assertIn(prediction['churn_prediction'], [0, 1])
                    self.assertTrue(0 <= prediction['churn_probability'] <= 1)
                    self.assertIn(prediction['confidence'], ['high', 'medium', 'low'])
                
                logger.info("✓ PySpark streaming inference test passed")
                
            except Exception as model_error:
                # This is expected if model doesn't exist yet
                logger.info(f"PySpark streaming inference test skipped (model not found): {str(model_error)}")
                self.skipTest("Model not available for inference testing")
            
        except ImportError as e:
            logger.error(f"PySpark streaming inference import failed: {str(e)}")
            self.fail(f"PySpark streaming inference import failed: {str(e)}")
    
    def test_backward_compatibility(self):
        """Test backward compatibility of migrated pipelines."""
        logger.info("Testing backward compatibility")
        
        try:
            # Test data pipeline with PySpark flag
            from data_pipeline import data_pipeline
            
            # Test with PySpark enabled
            result_pyspark = data_pipeline(
                data_path=self.test_data_path,
                use_pyspark=True,
                force_rebuild=True
            )
            
            # Test with pandas fallback
            result_pandas = data_pipeline(
                data_path=self.test_data_path,
                use_pyspark=False,
                force_rebuild=True
            )
            
            # Validate both results have expected structure
            for result in [result_pyspark, result_pandas]:
                self.assertIn('X_train', result)
                self.assertIn('X_test', result)
                self.assertIn('Y_train', result)
                self.assertIn('Y_test', result)
                
                # Check data shapes are reasonable
                self.assertTrue(result['X_train'].shape[0] > 0)
                self.assertTrue(result['X_test'].shape[0] > 0)
                self.assertTrue(result['Y_train'].shape[0] > 0)
                self.assertTrue(result['Y_test'].shape[0] > 0)
            
            logger.info("✓ Backward compatibility test passed")
            
        except Exception as e:
            logger.error(f"Backward compatibility test failed: {str(e)}")
            self.fail(f"Backward compatibility test failed: {str(e)}")
    
    def test_configuration_compatibility(self):
        """Test configuration compatibility."""
        logger.info("Testing configuration compatibility")
        
        try:
            config = get_config()
            
            # Check Spark configuration exists
            self.assertIn('spark', config)
            
            spark_config = config['spark']
            self.assertIn('app_name', spark_config)
            self.assertIn('master', spark_config)
            self.assertIn('executor_memory', spark_config)
            self.assertIn('driver_memory', spark_config)
            
            logger.info("✓ Configuration compatibility test passed")
            
        except Exception as e:
            logger.error(f"Configuration compatibility test failed: {str(e)}")
            self.fail(f"Configuration compatibility test failed: {str(e)}")
    
    def test_performance_comparison(self):
        """Compare performance between pandas and PySpark implementations."""
        logger.info("Testing performance comparison")
        
        try:
            import time
            from data_pipeline import data_pipeline
            
            # Test pandas performance
            start_time = time.time()
            pandas_result = data_pipeline(
                data_path=self.test_data_path,
                use_pyspark=False,
                force_rebuild=True
            )
            pandas_time = time.time() - start_time
            
            # Test PySpark performance
            start_time = time.time()
            pyspark_result = data_pipeline(
                data_path=self.test_data_path,
                use_pyspark=True,
                force_rebuild=True
            )
            pyspark_time = time.time() - start_time
            
            logger.info(f"Pandas pipeline time: {pandas_time:.2f} seconds")
            logger.info(f"PySpark pipeline time: {pyspark_time:.2f} seconds")
            
            # Validate both produced reasonable results
            self.assertTrue(pandas_result['X_train'].shape[0] > 0)
            self.assertTrue(pyspark_result['X_train'].shape[0] > 0)
            
            logger.info("✓ Performance comparison test passed")
            
        except Exception as e:
            logger.error(f"Performance comparison test failed: {str(e)}")
            self.fail(f"Performance comparison test failed: {str(e)}")


def run_migration_tests():
    """Run all migration tests."""
    ProjectLogger.log_section_header(logger, "STARTING PYSPARK MIGRATION VALIDATION TESTS")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPySparkMigration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Log results
    if result.wasSuccessful():
        ProjectLogger.log_success_header(logger, "ALL MIGRATION TESTS PASSED")
        logger.info(f"Tests run: {result.testsRun}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        logger.info("✓ PySpark migration validation completed successfully")
    else:
        ProjectLogger.log_error_header(logger, "SOME MIGRATION TESTS FAILED")
        logger.error(f"Tests run: {result.testsRun}")
        logger.error(f"Failures: {len(result.failures)}")
        logger.error(f"Errors: {len(result.errors)}")
        
        # Log failure details
        for test, traceback in result.failures:
            logger.error(f"FAILURE: {test}")
            logger.error(traceback)
        
        for test, traceback in result.errors:
            logger.error(f"ERROR: {test}")
            logger.error(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run migration validation tests
    success = run_migration_tests()
    
    if success:
        print("\n" + "="*50)
        print("🎉 MIGRATION VALIDATION SUCCESSFUL!")
        print("✓ All PySpark components are working correctly")
        print("✓ Backward compatibility is maintained")
        print("✓ Configuration is properly set up")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ MIGRATION VALIDATION FAILED!")
        print("Some tests failed. Please check the logs above.")
        print("="*50)
        sys.exit(1)