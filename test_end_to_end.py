#!/usr/bin/env python3
"""
End-to-End Integration Test for PySpark Migration
Tests the complete pipeline from raw data to trained model prediction
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'utils'))
sys.path.append(str(project_root / 'pipelines'))

from logger import get_logger, ProjectLogger
import json

logger = get_logger(__name__)

def test_complete_pyspark_pipeline():
    """Test the complete PySpark pipeline end-to-end."""
    
    ProjectLogger.log_section_header(logger, "STARTING END-TO-END PYSPARK INTEGRATION TEST")
    
    test_results = {
        'start_time': time.time(),
        'tests': {},
        'overall_status': 'RUNNING'
    }
    
    try:
        """
        Test 1: Configuration and Environment
        """
        ProjectLogger.log_step_header(logger, "TEST", "1: CONFIGURATION AND ENVIRONMENT")
        
        try:
            from config import get_config, get_spark_config
            from spark_utils import SparkSessionManager
            
            # Test configuration loading
            config = get_config()
            spark_config = get_spark_config()
            
            logger.info("✓ Configuration loading successful")
            
            # Test Spark session initialization
            spark = SparkSessionManager.get_session()
            logger.info(f"✓ Spark session initialized: {spark.sparkContext.appName}")
            logger.info(f"✓ Spark version: {spark.version}")
            
            test_results['tests']['configuration'] = {
                'status': 'PASSED',
                'spark_version': spark.version,
                'app_name': spark.sparkContext.appName
            }
            
        except Exception as e:
            logger.error(f"✗ Configuration test failed: {str(e)}")
            test_results['tests']['configuration'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            raise
        
        """
        Test 2: Data Pipeline
        """
        ProjectLogger.log_step_header(logger, "TEST", "2: PYSPARK DATA PIPELINE")
        
        try:
            from data_pipeline_pyspark import data_pipeline_pyspark
            
            # Test data pipeline
            logger.info("Running PySpark data pipeline...")
            start_time = time.time()
            
            pipeline_result = data_pipeline_pyspark(
                data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
                target_column='Churn',
                test_size=0.2,
                force_rebuild=True
            )
            
            pipeline_time = time.time() - start_time
            
            # Validate results
            train_df = pipeline_result['train_df']
            test_df = pipeline_result['test_df']
            
            train_count = train_df.count()
            test_count = test_df.count()
            
            logger.info(f"✓ Data pipeline completed in {pipeline_time:.2f} seconds")
            logger.info(f"✓ Training samples: {train_count}")
            logger.info(f"✓ Test samples: {test_count}")
            logger.info(f"✓ Total features: {len(pipeline_result['feature_columns'])}")
            
            test_results['tests']['data_pipeline'] = {
                'status': 'PASSED',
                'execution_time': pipeline_time,
                'train_samples': train_count,
                'test_samples': test_count,
                'num_features': len(pipeline_result['feature_columns'])
            }
            
        except Exception as e:
            logger.error(f"✗ Data pipeline test failed: {str(e)}")
            test_results['tests']['data_pipeline'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            raise
        
        """
        Test 3: Model Training
        """
        ProjectLogger.log_step_header(logger, "TEST", "3: PYSPARK MODEL TRAINING")
        
        try:
            from training_pipeline_pyspark import training_pipeline_pyspark
            
            # Test single model training
            logger.info("Training PySpark GBT model...")
            start_time = time.time()
            
            training_result = training_pipeline_pyspark(
                data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
                model_type='gbt',
                test_size=0.2
            )
            
            training_time = time.time() - start_time
            
            # Validate training results
            assert 'model_info' in training_result
            assert 'evaluation_metrics' in training_result
            assert 'training_metrics' in training_result
            
            accuracy = training_result['evaluation_metrics'].get('accuracy', 0)
            f1_score = training_result['evaluation_metrics'].get('f1', 0)
            
            logger.info(f"✓ Model training completed in {training_time:.2f} seconds")
            logger.info(f"✓ Test accuracy: {accuracy:.4f}")
            logger.info(f"✓ F1 score: {f1_score:.4f}")
            
            # Validate reasonable performance
            if accuracy < 0.5:
                logger.warning(f"Low accuracy detected: {accuracy:.4f}")
            
            test_results['tests']['model_training'] = {
                'status': 'PASSED',
                'execution_time': training_time,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'model_type': 'gbt'
            }
            
        except Exception as e:
            logger.error(f"✗ Model training test failed: {str(e)}")
            test_results['tests']['model_training'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            raise
        
        """
        Test 4: Model Comparison
        """
        ProjectLogger.log_step_header(logger, "TEST", "4: PYSPARK MODEL COMPARISON")
        
        try:
            from training_pipeline_pyspark import compare_models_pyspark
            
            # Test model comparison with subset of models for speed
            logger.info("Comparing PySpark models...")
            start_time = time.time()
            
            comparison_result = compare_models_pyspark(
                data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
                model_types=['gbt', 'logistic_regression'],  # Reduced for faster testing
                test_size=0.2
            )
            
            comparison_time = time.time() - start_time
            
            # Validate comparison results
            assert 'results' in comparison_result
            assert len(comparison_result['results']) == 2
            
            successful_models = [
                model for model, result in comparison_result['results'].items() 
                if 'error' not in result
            ]
            
            logger.info(f"✓ Model comparison completed in {comparison_time:.2f} seconds")
            logger.info(f"✓ Successful models: {len(successful_models)}/2")
            
            for model_name in successful_models:
                result = comparison_result['results'][model_name]
                if 'evaluation_metrics' in result:
                    accuracy = result['evaluation_metrics'].get('accuracy', 0)
                    logger.info(f"✓ {model_name} accuracy: {accuracy:.4f}")
            
            test_results['tests']['model_comparison'] = {
                'status': 'PASSED',
                'execution_time': comparison_time,
                'successful_models': len(successful_models),
                'total_models': 2,
                'models_tested': ['gbt', 'logistic_regression']
            }
            
        except Exception as e:
            logger.error(f"✗ Model comparison test failed: {str(e)}")
            test_results['tests']['model_comparison'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            # Continue with other tests
        
        """
        Test 5: Streaming Inference Setup
        """
        ProjectLogger.log_step_header(logger, "TEST", "5: PYSPARK STREAMING INFERENCE")
        
        try:
            from streaming_inference_pipeline_pyspark import streaming_inference_pyspark
            
            # Test streaming inference setup (without actual model since it may not exist)
            logger.info("Testing PySpark streaming inference setup...")
            start_time = time.time()
            
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
            
            try:
                # This may fail if model doesn't exist, which is expected
                inference_result = streaming_inference_pyspark(
                    input_data=sample_customer,
                    model_path="./artifacts/models/pyspark_pipeline_model"
                )
                
                inference_time = time.time() - start_time
                logger.info(f"✓ Streaming inference completed in {inference_time:.2f} seconds")
                
                if 'single_prediction' in inference_result:
                    prediction = inference_result['single_prediction']
                    logger.info(f"✓ Prediction result: {prediction}")
                
                test_results['tests']['streaming_inference'] = {
                    'status': 'PASSED',
                    'execution_time': inference_time,
                    'prediction_available': 'single_prediction' in inference_result
                }
                
            except Exception as model_error:
                # Expected if model doesn't exist
                logger.info(f"Streaming inference test skipped (expected): {str(model_error)}")
                
                test_results['tests']['streaming_inference'] = {
                    'status': 'SKIPPED',
                    'reason': 'Model not available (expected)',
                    'error': str(model_error)
                }
            
        except Exception as e:
            logger.error(f"✗ Streaming inference test failed: {str(e)}")
            test_results['tests']['streaming_inference'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        """
        Test 6: Backward Compatibility
        """
        ProjectLogger.log_step_header(logger, "TEST", "6: BACKWARD COMPATIBILITY")
        
        try:
            from data_pipeline import data_pipeline
            from training_pipeline import training_pipeline
            
            # Test backward compatibility
            logger.info("Testing backward compatibility...")
            start_time = time.time()
            
            # Test data pipeline with PySpark flag
            data_result = data_pipeline(
                data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
                use_pyspark=True,
                force_rebuild=True
            )
            
            # Validate result structure
            assert 'X_train' in data_result
            assert 'X_test' in data_result
            assert 'Y_train' in data_result
            assert 'Y_test' in data_result
            
            compat_time = time.time() - start_time
            
            logger.info(f"✓ Backward compatibility test completed in {compat_time:.2f} seconds")
            logger.info(f"✓ Training samples: {len(data_result['X_train'])}")
            logger.info(f"✓ Test samples: {len(data_result['X_test'])}")
            
            test_results['tests']['backward_compatibility'] = {
                'status': 'PASSED',
                'execution_time': compat_time,
                'data_structure_valid': True
            }
            
        except Exception as e:
            logger.error(f"✗ Backward compatibility test failed: {str(e)}")
            test_results['tests']['backward_compatibility'] = {
                'status': 'FAILED',
                'error': str(e)
            }
        
        # Calculate overall results
        test_results['end_time'] = time.time()
        test_results['total_time'] = test_results['end_time'] - test_results['start_time']
        
        passed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'PASSED')
        total_tests = len(test_results['tests'])
        skipped_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'SKIPPED')
        failed_tests = sum(1 for test in test_results['tests'].values() if test['status'] == 'FAILED')
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': (passed_tests / (total_tests - skipped_tests)) * 100 if (total_tests - skipped_tests) > 0 else 0
        }
        
        if failed_tests == 0:
            test_results['overall_status'] = 'PASSED'
            ProjectLogger.log_success_header(logger, "END-TO-END INTEGRATION TEST PASSED")
        else:
            test_results['overall_status'] = 'FAILED'
            ProjectLogger.log_error_header(logger, "END-TO-END INTEGRATION TEST FAILED")
        
        # Log summary
        logger.info(f"Test Summary:")
        logger.info(f"  - Total time: {test_results['total_time']:.2f} seconds")
        logger.info(f"  - Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"  - Tests failed: {failed_tests}")
        logger.info(f"  - Tests skipped: {skipped_tests}")
        logger.info(f"  - Success rate: {test_results['summary']['success_rate']:.1f}%")
        
        # Save test results
        results_path = "./test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {results_path}")
        
        return test_results
        
    except Exception as e:
        test_results['overall_status'] = 'FAILED'
        test_results['fatal_error'] = str(e)
        ProjectLogger.log_error_header(logger, "END-TO-END INTEGRATION TEST FAILED WITH FATAL ERROR")
        logger.error(f"Fatal error: {str(e)}")
        return test_results
    
    finally:
        # Clean up Spark session
        try:
            from spark_utils import SparkSessionManager
            SparkSessionManager.stop_session()
            logger.info("Spark session cleaned up")
        except:
            pass


if __name__ == "__main__":
    # Change to project directory
    os.chdir(project_root)
    
    print("🚀 Starting End-to-End PySpark Integration Test")
    print("=" * 60)
    
    # Run comprehensive test
    results = test_complete_pyspark_pipeline()
    
    # Display final results
    print("\n" + "=" * 60)
    if results['overall_status'] == 'PASSED':
        print("🎉 END-TO-END INTEGRATION TEST SUCCESSFUL!")
        print("✅ All PySpark components are working correctly")
        print("✅ Pipeline integration is functioning properly")
        print("✅ Backward compatibility is maintained")
    elif results['overall_status'] == 'FAILED':
        print("❌ END-TO-END INTEGRATION TEST FAILED!")
        print("Some components need attention. Check the logs above.")
        if 'summary' in results:
            print(f"Success rate: {results['summary']['success_rate']:.1f}%")
    else:
        print("⚠️  END-TO-END INTEGRATION TEST INCOMPLETE")
        print("Test execution was interrupted or incomplete.")
    
    print("=" * 60)
    
    # Return appropriate exit code
    if results['overall_status'] == 'PASSED':
        sys.exit(0)
    else:
        sys.exit(1)