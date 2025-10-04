# PySpark Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Performance Tuning](#performance-tuning)
8. [Deployment Guide](#deployment-guide)
9. [Troubleshooting](#troubleshooting)
10. [Migration from Pandas](#migration-from-pandas)

---

## Overview

The Telco Customer Churn Prediction system has been successfully migrated from pandas + scikit-learn to **PySpark ML** for distributed computing capabilities. This enables the system to:

- **Scale horizontally** across multiple nodes
- **Process large datasets** efficiently with distributed computing
- **Handle real-time streaming** data with Structured Streaming
- **Maintain backward compatibility** with the original pandas implementation

### Key Features

✅ **Distributed Data Processing**: Handle datasets from GB to TB scale  
✅ **ML Pipeline**: End-to-end machine learning with PySpark ML  
✅ **Real-time Inference**: Streaming predictions with fault tolerance  
✅ **Model Comparison**: Multiple algorithms with automated evaluation  
✅ **MLflow Integration**: Experiment tracking and model registry  
✅ **Backward Compatibility**: Seamless fallback to pandas implementation  

---

## Quick Start

### 1. Basic Data Pipeline
```python
from pipelines.data_pipeline import data_pipeline

# Process data with PySpark
result = data_pipeline(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    use_pyspark=True,
    force_rebuild=True
)

print(f"Training samples: {len(result['X_train'])}")
print(f"Test samples: {len(result['X_test'])}")
```

### 2. Train a Model
```python
from pipelines.training_pipeline import training_pipeline

# Train model with PySpark ML
models = training_pipeline(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    use_pyspark=True
)

# Get results
accuracy = models['evaluation_metrics']['accuracy']
print(f"Model accuracy: {accuracy:.4f}")
```

### 3. Make Predictions
```python
from pipelines.streaming_inference_pipeline import streaming_inference

# Sample customer data
customer = {
    'gender': 'Male',
    'tenure': 12.0,
    'MonthlyCharges': 50.0,
    # ... other features
}

# Get prediction
prediction = streaming_inference(
    use_pyspark=True,
    input_data=customer
)
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Java 8 or 11 (required for PySpark)
- Minimum 4GB RAM (8GB+ recommended)

### Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify PySpark installation
python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
```

### Environment Setup
```bash
# Set Java home (if needed)
export JAVA_HOME=/path/to/java

# Set Spark home (optional)
export SPARK_HOME=/path/to/spark

# Configure memory (optional)
export SPARK_DRIVER_MEMORY=2g
export SPARK_EXECUTOR_MEMORY=2g
```

---

## Configuration

### Spark Configuration (`config.yaml`)
```yaml
spark:
  app_name: "TelcoCustomerChurnPrediction"
  master: "local[*]"  # Use all available cores
  executor_memory: "2g"
  driver_memory: "1g"
  max_result_size: "1g"
  sql_adaptive_enabled: true
  sql_adaptive_coalesce_partitions_enabled: true
```

### Common Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `master` | Cluster manager | `local[*]` | `local[*]`, `yarn`, `k8s://...` |
| `executor_memory` | Memory per executor | `2g` | `1g`, `2g`, `4g`, etc. |
| `driver_memory` | Driver memory | `1g` | `512m`, `1g`, `2g`, etc. |
| `max_result_size` | Max result size | `1g` | `512m`, `1g`, `2g`, etc. |

### MLflow Configuration
```yaml
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "telco_pyspark_experiment"
  autolog: true
  model_registry_name: "pyspark_churn_model"
```

---

## Usage Examples

### Data Pipeline Examples

#### Basic Data Processing
```python
from pipelines.data_pipeline_pyspark import data_pipeline_pyspark

# Full data pipeline
result = data_pipeline_pyspark(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    target_column='Churn',
    test_size=0.2,
    force_rebuild=True
)

# Access processed data
train_df = result['train_df']  # PySpark DataFrame
test_df = result['test_df']    # PySpark DataFrame
feature_columns = result['feature_columns']

# Display data info
print(f"Training samples: {train_df.count()}")
print(f"Test samples: {test_df.count()}")
print(f"Features: {len(feature_columns)}")
```

#### Using Existing Processed Data
```python
# Skip preprocessing if data exists
result = data_pipeline_pyspark(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    force_rebuild=False  # Will use existing processed data
)
```

### Training Examples

#### Single Model Training
```python
from pipelines.training_pipeline_pyspark import training_pipeline_pyspark

# Train Gradient Boosted Trees
gbt_result = training_pipeline_pyspark(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    model_type='gbt',
    model_params={
        'maxDepth': 5,
        'maxIter': 20,
        'featuresCol': 'features',
        'labelCol': 'Churn'
    }
)

# Train Random Forest
rf_result = training_pipeline_pyspark(
    model_type='random_forest',
    model_params={
        'numTrees': 100,
        'maxDepth': 5,
        'featuresCol': 'features',
        'labelCol': 'Churn'
    }
)
```

#### Model Comparison
```python
from pipelines.training_pipeline_pyspark import compare_models_pyspark

# Compare multiple models
comparison = compare_models_pyspark(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    model_types=['gbt', 'random_forest', 'logistic_regression', 'decision_tree'],
    test_size=0.2
)

# View results
for model_name, result in comparison['results'].items():
    if 'evaluation_metrics' in result:
        accuracy = result['evaluation_metrics']['accuracy']
        f1 = result['evaluation_metrics']['f1']
        print(f"{model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
```

### Inference Examples

#### Single Customer Prediction
```python
from pipelines.streaming_inference_pipeline_pyspark import streaming_inference_pyspark

# Customer data
customer_data = {
    'gender': 'Female',
    'SeniorCitizen': '0',
    'Partner': 'No',
    'Dependents': 'Yes',
    'tenure': 24.0,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Credit card (automatic)',
    'MonthlyCharges': 85.0,
    'TotalCharges': 2040.0
}

# Make prediction
result = streaming_inference_pyspark(
    model_path="./artifacts/models/pyspark_pipeline_model",
    input_data=customer_data
)

prediction = result['single_prediction']
print(f"Churn prediction: {prediction['churn_prediction']}")
print(f"Churn probability: {prediction['churn_probability']:.4f}")
print(f"Confidence: {prediction['confidence']}")
```

#### Batch Prediction
```python
# Batch prediction on CSV file
batch_result = streaming_inference_pyspark(
    model_path="./artifacts/models/pyspark_pipeline_model",
    batch_data_path="./data/new_customers.csv"
)

print(f"Processed {batch_result['batch_predictions']} records")
print(f"Results saved to: {batch_result['batch_output_path']}")
```

#### Streaming Prediction Setup
```python
# Set up real-time streaming predictions
streaming_query = streaming_inference_pyspark(
    model_path="./artifacts/models/pyspark_pipeline_model",
    stream_input_path="./streaming/input/",
    stream_output_path="./streaming/output/",
    stream_checkpoint_path="./streaming/checkpoints/"
)

# Monitor streaming
query = streaming_query['streaming_query']
query.awaitTermination()
```

### Backward Compatibility Examples

#### Using PySpark with Legacy Interface
```python
from pipelines.data_pipeline import data_pipeline
from pipelines.training_pipeline import training_pipeline

# Data pipeline with PySpark backend
data_result = data_pipeline(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    use_pyspark=True,  # Use PySpark implementation
    force_rebuild=True
)

# Training pipeline with PySpark backend
train_result = training_pipeline(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    use_pyspark=True  # Use PySpark implementation
)
```

#### Fallback to Pandas
```python
# Fallback to pandas implementation
data_result = data_pipeline(
    use_pyspark=False  # Use original pandas implementation
)
```

---

## API Reference

### Core Classes

#### `SparkSessionManager`
Singleton class for managing Spark sessions.

```python
from spark_utils import SparkSessionManager

# Get Spark session
spark = SparkSessionManager.get_session()

# Stop session
SparkSessionManager.stop_session()

# Get session info
info = SparkSessionManager.get_session_info()
```

#### `MLflowTracker`
Enhanced MLflow tracking with PySpark support.

```python
from mlflow_utils import MLflowTracker

tracker = MLflowTracker()

# Start run
with tracker.start_run(run_name="PySpark Training"):
    # Log PySpark data pipeline
    tracker.log_pyspark_data_pipeline_metrics(data_info)
    
    # Log PySpark model
    tracker.log_pyspark_model(model, metrics, params)
```

### Pipeline Functions

#### `data_pipeline_pyspark()`
Complete data preprocessing pipeline using PySpark.

**Parameters:**
- `data_path` (str): Path to raw data file
- `target_column` (str): Name of target column (default: 'Churn')
- `test_size` (float): Test split ratio (default: 0.2)
- `force_rebuild` (bool): Force rebuild processed data (default: False)

**Returns:**
- `dict`: Contains 'train_df', 'test_df', 'feature_columns'

#### `training_pipeline_pyspark()`
Model training pipeline using PySpark ML.

**Parameters:**
- `data_path` (str): Path to data file
- `model_type` (str): Model type ('gbt', 'random_forest', 'logistic_regression', 'decision_tree')
- `model_params` (dict): Model hyperparameters
- `test_size` (float): Test split ratio
- `target_column` (str): Target column name

**Returns:**
- `dict`: Contains model info, metrics, and evaluation results

#### `streaming_inference_pyspark()`
Real-time inference pipeline using PySpark.

**Parameters:**
- `model_path` (str): Path to trained model
- `input_data` (dict): Single customer data for prediction
- `batch_data_path` (str): Path to batch data file
- `stream_input_path` (str): Streaming input directory
- `stream_output_path` (str): Streaming output directory

**Returns:**
- `dict`: Prediction results and streaming query status

---

## Performance Tuning

### Memory Configuration

#### For Small Datasets (< 1GB)
```yaml
spark:
  executor_memory: "1g"
  driver_memory: "512m"
  max_result_size: "512m"
```

#### For Medium Datasets (1-10GB)
```yaml
spark:
  executor_memory: "4g"
  driver_memory: "2g"
  max_result_size: "2g"
```

#### For Large Datasets (> 10GB)
```yaml
spark:
  executor_memory: "8g"
  driver_memory: "4g"
  max_result_size: "4g"
```

### Optimization Settings

#### Enable Adaptive Query Execution
```yaml
spark:
  sql_adaptive_enabled: true
  sql_adaptive_coalesce_partitions_enabled: true
  sql_adaptive_skew_join_enabled: true
```

#### Optimize Shuffle Operations
```yaml
spark:
  shuffle_partitions: 200  # Adjust based on data size
  serializer: "org.apache.spark.serializer.KryoSerializer"
```

### Performance Best Practices

1. **Cache DataFrames**: Cache frequently accessed DataFrames
   ```python
   df.cache()
   df.count()  # Trigger caching
   ```

2. **Optimize Partitioning**: Repartition data for better parallelism
   ```python
   df = df.repartition(200)  # Adjust number based on cluster size
   ```

3. **Use Parquet Format**: Store processed data in Parquet for better performance
   ```python
   df.write.mode("overwrite").parquet("path/to/data.parquet")
   ```

4. **Broadcast Small DataFrames**: Use broadcast joins for small lookup tables
   ```python
   from pyspark.sql.functions import broadcast
   result = large_df.join(broadcast(small_df), "key")
   ```

---

## Deployment Guide

### Local Development
```bash
# Single machine deployment
python pipelines/training_pipeline_pyspark.py
```

### Standalone Cluster
```bash
# Start Spark standalone cluster
$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/start-worker.sh spark://master:7077

# Submit job to cluster
spark-submit \
  --master spark://master:7077 \
  --executor-memory 2g \
  --driver-memory 1g \
  pipelines/training_pipeline_pyspark.py
```

### YARN Cluster
```bash
# Submit to YARN cluster
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 4g \
  --driver-memory 2g \
  --num-executors 10 \
  pipelines/training_pipeline_pyspark.py
```

### Kubernetes
```yaml
# kubernetes-deployment.yaml
apiVersion: v1
kind: Pod
metadata:
  name: spark-driver
spec:
  containers:
  - name: spark-driver
    image: bitnami/spark:latest
    command: ["/opt/bitnami/spark/bin/spark-submit"]
    args:
      - --master
      - k8s://https://kubernetes.default.svc:443
      - --deploy-mode
      - client
      - pipelines/training_pipeline_pyspark.py
```

### Cloud Deployment

#### AWS EMR
```bash
# Create EMR cluster
aws emr create-cluster \
  --name "Telco-Churn-PySpark" \
  --release-label emr-6.4.0 \
  --applications Name=Spark \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
                   InstanceGroupType=CORE,InstanceCount=2,InstanceType=m5.xlarge
```

#### Azure HDInsight
```bash
# Create HDInsight cluster
az hdinsight create \
  --name telco-churn-cluster \
  --resource-group myResourceGroup \
  --type spark \
  --cluster-login-password myPassword \
  --ssh-password myPassword
```

#### Google Cloud Dataproc
```bash
# Create Dataproc cluster
gcloud dataproc clusters create telco-churn-cluster \
  --enable-autoscaling \
  --num-workers 2 \
  --max-workers 10 \
  --worker-machine-type n1-standard-4
```

---

## Troubleshooting

### Common Issues

#### 1. Java Version Issues
```bash
# Check Java version
java -version

# Set correct Java home
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
```

#### 2. Memory Issues
```
Error: OutOfMemoryError
```
**Solution:** Increase memory allocation in `config.yaml`
```yaml
spark:
  executor_memory: "4g"  # Increase from 2g
  driver_memory: "2g"    # Increase from 1g
```

#### 3. Serialization Issues
```
Error: Task not serializable
```
**Solution:** Ensure all objects used in transformations are serializable
```python
# Instead of using class methods in transformations
def transform_func(row):
    return row * 2

df.map(transform_func)  # This works
```

#### 4. Partition Issues
```
Warning: Large number of small partitions
```
**Solution:** Repartition the DataFrame
```python
df = df.repartition(200)  # Reduce number of partitions
```

#### 5. MLflow Logging Issues
```
Error: Failed to log model to MLflow
```
**Solution:** Check MLflow configuration and model format
```python
# Ensure model is PySpark ML model
if hasattr(model, 'stages'):
    mlflow.spark.log_model(model, "pyspark_model")
```

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger("py4j").setLevel(logging.DEBUG)
logging.getLogger("pyspark").setLevel(logging.DEBUG)
```

Monitor Spark UI:
- Local: http://localhost:4040
- Cluster: Check cluster manager UI

---

## Migration from Pandas

### Key Differences

| Aspect | Pandas | PySpark |
|--------|--------|---------|
| **Data Structure** | DataFrame (in-memory) | DataFrame (distributed) |
| **Execution** | Eager | Lazy |
| **Memory** | Single machine | Distributed |
| **API** | Similar to SQL/NumPy | Similar to SQL |
| **Scaling** | Vertical only | Horizontal |

### Migration Checklist

✅ **Code Changes**
- Replace pandas imports with PySpark
- Update DataFrame operations
- Use PySpark ML instead of scikit-learn
- Handle lazy evaluation

✅ **Data Changes**
- Convert to Parquet format for better performance
- Partition large datasets appropriately
- Consider data locality

✅ **Configuration**
- Set up Spark configuration
- Configure memory allocation
- Set up cluster (if needed)

✅ **Testing**
- Test with small datasets first
- Validate results match pandas implementation
- Performance test with larger datasets

### Step-by-step Migration

#### 1. Update Dependencies
```bash
pip install pyspark==3.4.0
```

#### 2. Update Code
```python
# Before (Pandas)
import pandas as pd
df = pd.read_csv("data.csv")
df = df.dropna()

# After (PySpark)
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("MyApp").getOrCreate()
df = spark.read.option("header", "true").csv("data.csv")
df = df.dropna()
```

#### 3. Update ML Code
```python
# Before (scikit-learn)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# After (PySpark ML)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
rf = RandomForestClassifier(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[rf])
model = pipeline.fit(train_df)
```

#### 4. Test and Validate
```python
# Run end-to-end test
python test_end_to_end.py
```

---

## Additional Resources

### Documentation
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [PySpark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Examples
- [PySpark Examples](https://github.com/apache/spark/tree/master/examples/src/main/python)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)

### Community
- [Stack Overflow: PySpark](https://stackoverflow.com/questions/tagged/pyspark)
- [Spark User Mailing List](https://spark.apache.org/community.html)

---

**🚀 Ready to scale your machine learning with PySpark!**

For questions or issues, please check the troubleshooting section or refer to the community resources above.