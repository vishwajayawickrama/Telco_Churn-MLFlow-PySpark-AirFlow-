import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from mlflow_utils import MLflowTracker, create_mlflow_run_tags
from model_inference import ModelInference
from logger import get_logger, ProjectLogger, log_exceptions
import mlflow
import time

# Initialize logger
logger = get_logger(__name__)

# Initialize model inference with error handling
try:
    ProjectLogger.log_section_header(logger, "INITIALIZING STREAMING INFERENCE PIPELINE")
    logger.info("Starting model inference initialization for streaming pipeline")
    
    model_path = 'artifacts/models/telco_customer_churn_prediction.joblib'
    logger.info(f"Loading model from: {model_path}")
    
    inference = ModelInference(model_path)
    logger.info("Model inference instance created successfully")
    
    ProjectLogger.log_success_header(logger, "STREAMING INFERENCE PIPELINE INITIALIZED")
    
except Exception as e:
    ProjectLogger.log_error_header(logger, "STREAMING INFERENCE PIPELINE INITIALIZATION FAILED")
    logger.error(f"Failed to initialize streaming inference pipeline: {str(e)}")
    logger.error("Pipeline cannot proceed without valid model instance")
    raise

@log_exceptions(logger)
def streaming_inference(inference, data):
    """
    Perform streaming inference on incoming data.

    Args:
        inference (ModelInference): An instance of the ModelInference class.
        data (iterable): An iterable that yields data points for inference.
        
    Returns:
        dict: Prediction results from the model
    """
    ProjectLogger.log_step_header(logger, "STEP", "STREAMING INFERENCE EXECUTION")
    
    try:
        logger.info("Starting streaming inference process")
        
        # Start MLflow tracking
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags(
                                            'streaming_inference', 
                                            {
                                                'inference_type': 'single_record',
                                                'model_type': 'XGBoost'
                                            }
                                        )
        run = mlflow_tracker.start_run(run_name='streaming_inference', tags=run_tags)
        logger.info("Inference tracking run started")
        
        # Validate inputs
        if inference is None:
            raise ValueError("ModelInference instance is None")
        
        if data is None:
            raise ValueError("Input data is None")
        
        logger.info("Input validation completed successfully")
        logger.debug(f"Input data type: {type(data)}")
        
        # Load encoders
        try:
            logger.info("Loading encoders for data preprocessing")
            encoder_path = 'artifacts/encode'
            logger.debug(f"Encoder path: {encoder_path}")
            
            inference.load_encoders(encoder_path)
            logger.info("Encoders loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading encoders: {str(e)}")
            raise
        
        # Perform prediction
        try:
            logger.info("Executing model prediction")
            start_time = time.time()

            pred = inference.predict(data)

            inference_time = time.time() - start_time

            if pred is None:
                raise ValueError("Model prediction returned None")
            
                    # Log inference metrics to MLflow
            mlflow.log_metrics({
                                'inference_time_ms': inference_time,
                                'churn_probability': float(pred['Confidence'].replace('%', '')) / 100,
                                'predicted_class': 1 if pred['Status'] == 'Churn' else 0
                              })
            
            mlflow.log_params({f'input_{k}': v for k, v in data.items()})
            
            logger.info("Model prediction completed successfully")
            logger.debug(f"Prediction result: {pred}")
            
            ProjectLogger.log_success_header(logger, "STREAMING INFERENCE COMPLETED")
            return pred
            
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            raise
            
    except ValueError as e:
        ProjectLogger.log_error_header(logger, "STREAMING INFERENCE VALIDATION ERROR")
        logger.error(f"Validation error: {str(e)}")
        raise
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "STREAMING INFERENCE FAILED")
        logger.error(f"Unexpected error in streaming inference: {str(e)}")
        raise
    finally:
        mlflow_tracker.end_run()


if __name__ == "__main__":
    try:
        ProjectLogger.log_section_header(logger, "EXECUTING STREAMING INFERENCE DEMO")
        logger.info("Starting streaming inference demonstration")
        
        # Sample data for demonstration (mock customer in actual dataset format)
        data = {
            "customerID": "8472-KMTXZ",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes", 
            "Dependents": "No",
            "tenure": 18,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.75,
            "TotalCharges": 1615.5,
        }
        
        logger.info("Sample data prepared for inference")
        logger.debug(f"Sample data: {data}")
        
        # Validate sample data
        required_fields = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService", "PaymentMethod"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"Missing required fields in sample data: {missing_fields}")
        
        # Execute streaming inference
        logger.info("Executing streaming inference with sample data")
        pred = streaming_inference(inference, data)
        
        # Validate and display results
        if pred is None:
            logger.error("Prediction result is None")
            raise ValueError("Invalid prediction result")
        
        logger.info("Streaming inference completed successfully")
        logger.info(f"Prediction result: {pred}")
        
        # Display results
        print("=" * 50)
        print("STREAMING INFERENCE RESULT")
        print("=" * 50)
        print(pred)
        print("=" * 50)
        
        ProjectLogger.log_success_header(logger, "STREAMING INFERENCE DEMO COMPLETED")
        
    except KeyError as e:
        ProjectLogger.log_error_header(logger, "DATA VALIDATION ERROR")
        logger.error(f"Missing required data field: {str(e)}")
        print(f"Error: Missing required data field - {str(e)}")
        
    except ValueError as e:
        ProjectLogger.log_error_header(logger, "VALUE ERROR")
        logger.error(f"Invalid data value: {str(e)}")
        print(f"Error: Invalid data value - {str(e)}")
        
    except Exception as e:
        ProjectLogger.log_error_header(logger, "STREAMING INFERENCE DEMO FAILED")
        logger.error(f"Unexpected error in demo execution: {str(e)}")
        print(f"Error: Streaming inference demo failed - {str(e)}")
        raise

