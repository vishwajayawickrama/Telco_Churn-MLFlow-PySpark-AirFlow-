# Telco Customer Churn Prediction - PySpark Implementation

![PySpark](https://img.shields.io/badge/PySpark-3.4.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.6.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Overview

A production-ready customer churn prediction system built with **PySpark ML** for distributed computing and scalable machine learning. This implementation provides enterprise-grade features including distributed data processing, real-time inference, and comprehensive MLflow integration.

### ✨ Key Features

- 🔥 **Distributed Computing**: Scale from single machine to cluster deployment
- 🤖 **Advanced ML Pipeline**: End-to-end PySpark ML with 4 algorithms
- 📊 **Real-time Streaming**: Live predictions with Structured Streaming
- 📈 **MLflow Integration**: Experiment tracking and model registry
- 🔄 **Backward Compatible**: Seamless fallback to pandas implementation
- 🛠️ **Production Ready**: Comprehensive logging, error handling, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Data       │───▶│  PySpark Data    │───▶│  Processed Data │
│  (CSV/Parquet)  │    │  Pipeline        │    │  (Parquet)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Model Registry │◀───│  PySpark ML      │───▶│  Trained Models │
│  (MLflow)       │    │  Training        │    │  (Pipeline)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Predictions    │◀───│  Streaming       │◀───│  Live Data      │
│  (Real-time)    │    │  Inference       │    │  (JSON/Kafka)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Required
- Python 3.8+
- Java 8 or 11
- Minimum 4GB RAM (8GB+ recommended)

# Optional (for cluster deployment)
- Apache Spark 3.4.0+
- Hadoop (for YARN deployment)
- Kubernetes (for K8s deployment)
```

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd telco-churn-pyspark

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pyspark; print(f'PySpark {pyspark.__version__} installed successfully')"
```

### 3. Configuration

```bash
# Copy example configuration
cp config.example.yaml config.yaml

# Edit configuration (optional)
nano config.yaml
```

### 4. Run Training Pipeline

```python
from pipelines.training_pipeline import training_pipeline

# Train model with PySpark
result = training_pipeline(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    use_pyspark=True
)

print(f"Model accuracy: {result['evaluation_metrics']['accuracy']:.4f}")
```

## 📊 Usage Examples

### Data Processing

```python
from pipelines.data_pipeline import data_pipeline

# Process data with PySpark
data_result = data_pipeline(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    use_pyspark=True,
    force_rebuild=True
)

print(f"Training samples: {len(data_result['X_train'])}")
print(f"Test samples: {len(data_result['X_test'])}")
```

### Model Training & Comparison

```python
from pipelines.training_pipeline import compare_models_pyspark

# Compare multiple models
comparison = compare_models_pyspark(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    model_types=['gbt', 'random_forest', 'logistic_regression'],
    test_size=0.2
)

# View results
for model, result in comparison['results'].items():
    if 'evaluation_metrics' in result:
        print(f"{model}: {result['evaluation_metrics']['accuracy']:.4f}")
```

### Real-time Inference

```python
from pipelines.streaming_inference_pipeline import streaming_inference

# Customer data
customer = {
    'gender': 'Female',
    'tenure': 24,
    'MonthlyCharges': 85.0,
    'PaymentMethod': 'Credit card (automatic)',
    # ... other features
}

# Get prediction
prediction = streaming_inference(
    use_pyspark=True,
    input_data=customer
)

print(f"Churn probability: {prediction['single_prediction']['churn_probability']:.4f}")
```

## 🔧 Configuration

### Spark Configuration (`config.yaml`)

```yaml
spark:
  app_name: "TelcoCustomerChurnPrediction"
  master: "local[*]"              # Use all cores
  executor_memory: "2g"           # Memory per executor
  driver_memory: "1g"             # Driver memory
  max_result_size: "1g"           # Max result size
  sql_adaptive_enabled: true      # Enable AQE
  sql_adaptive_coalesce_partitions_enabled: true
```

### MLflow Configuration

```yaml
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "telco_pyspark_experiment"
  autolog: true
  model_registry_name: "pyspark_churn_model"
  run_name_prefix: "pyspark_run"
```

### Model Configuration

```yaml
model:
  algorithms:
    gbt:
      maxDepth: 5
      maxIter: 20
    random_forest:
      numTrees: 100
      maxDepth: 5
    logistic_regression:
      maxIter: 100
      regParam: 0.01
```

## 🚀 Deployment Options

### Local Development

```bash
# Single machine with all cores
python pipelines/training_pipeline.py
```

### Standalone Cluster

```bash
# Start Spark cluster
$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/start-workers.sh

# Submit job
spark-submit \
  --master spark://master:7077 \
  --executor-memory 4g \
  --driver-memory 2g \
  pipelines/training_pipeline_pyspark.py
```

### YARN Cluster

```bash
# Submit to YARN
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 4g \
  --num-executors 10 \
  pipelines/training_pipeline_pyspark.py
```

### Cloud Deployment

#### AWS EMR
```bash
aws emr create-cluster \
  --name "Telco-Churn-PySpark" \
  --release-label emr-6.4.0 \
  --applications Name=Spark \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge
```

#### Azure HDInsight
```bash
az hdinsight create \
  --name telco-churn-cluster \
  --type spark \
  --cluster-login-password myPassword
```

#### Google Cloud Dataproc
```bash
gcloud dataproc clusters create telco-churn-cluster \
  --enable-autoscaling \
  --max-workers 10
```

## 📈 Performance Optimization

### Memory Tuning

| Dataset Size | Executor Memory | Driver Memory | Max Result Size |
|--------------|----------------|---------------|-----------------|
| < 1GB        | 1g             | 512m          | 512m            |
| 1-10GB       | 4g             | 2g            | 2g              |
| 10-100GB     | 8g             | 4g            | 4g              |
| > 100GB      | 16g            | 8g            | 8g              |

### Best Practices

1. **Cache DataFrames**: Cache frequently accessed data
2. **Use Parquet**: Store data in columnar format
3. **Optimize Partitions**: Balance parallelism vs overhead
4. **Broadcast Joins**: Use for small lookup tables
5. **Enable AQE**: Adaptive Query Execution for optimization

## 🧪 Testing

### Run All Tests

```bash
# Complete integration test
python test_end_to_end.py

# Migration validation
python test_migration.py

# Demonstration
python migration_demo.py
```

### Test Individual Components

```python
# Test data pipeline
from pipelines.data_pipeline import data_pipeline_pyspark
result = data_pipeline_pyspark(force_rebuild=True)

# Test model training
from pipelines.training_pipeline import training_pipeline_pyspark
model = training_pipeline_pyspark(model_type='gbt')

# Test inference
from pipelines.streaming_inference_pipeline import streaming_inference_pyspark
prediction = streaming_inference_pyspark(input_data=sample_customer)
```

## 📊 Monitoring & Observability

### Spark UI
- **Local**: http://localhost:4040
- **Cluster**: Check cluster manager UI

### MLflow UI
```bash
# Start MLflow server
mlflow ui --port 5000

# Access at http://localhost:5000
```

### Logging
All components include comprehensive logging:
- **Data Pipeline**: Processing steps and statistics
- **Training**: Model metrics and performance
- **Inference**: Prediction results and timing
- **Errors**: Detailed error tracking and recovery

## 🔍 Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Increase memory allocation
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=8g
```

#### Java Issues
```bash
# Set Java home
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

#### Serialization Errors
Ensure all functions used in Spark transformations are serializable.

#### Performance Issues
- Check Spark UI for job details
- Optimize partition sizes
- Enable adaptive query execution

## 📁 Project Structure

```
├── src/                          # Core PySpark modules
│   ├── data_ingestion.py         # PySpark data loading
│   ├── feature_encoding.py       # PySpark ML encoders
│   ├── model_building.py         # PySpark ML models
│   └── ...
├── pipelines/                    # Pipeline implementations
│   ├── data_pipeline.py  # PySpark data pipeline
│   ├── training_pipeline.py # PySpark training
│   └── streaming_inference_pipeline.py # Real-time inference
├── utils/                        # Utilities
│   ├── spark_utils.py           # Spark session management
│   ├── mlflow_utils.py          # MLflow PySpark support
│   └── config.py                # Configuration management
├── data/                         # Data directories
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data (Parquet)
├── artifacts/                    # Model artifacts
│   ├── models/                  # Trained models
│   └── encode/                  # Encoders
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Apache Spark** for distributed computing framework
- **MLflow** for experiment tracking and model registry
- **PySpark ML** for machine learning capabilities
- **Community** for continuous support and contributions

---

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: See [PYSPARK_GUIDE.md](PYSPARK_GUIDE.md) for detailed guide
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

---

**🚀 Ready to scale your machine learning with PySpark!**

*Built with ❤️ for enterprise-scale machine learning*