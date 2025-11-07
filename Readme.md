# Telco Customer Churn Prediction - PySpark Implementation

![PySpark](https://img.shields.io/badge/PySpark-3.4.0+-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.0.0+-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.7.0-red)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Pandas](https://img.shields.io/badge/Pandas-1.5.0+-purple)
![NumPy](https://img.shields.io/badge/NumPy-1.21.0+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📚 Technology Stack

### Core Technologies
- **Apache Spark (PySpark 3.4.0+)**: Distributed data processing and ML
- **Apache Airflow (2.7.0)**: Workflow orchestration and scheduling
- **MLflow (2.0.0+)**: Experiment tracking and model registry
- **Python (3.8+)**: Primary programming language

### Data Processing & ML
- **PySpark ML**: Machine learning algorithms (GBT, Random Forest, Logistic Regression, Decision Tree)
- **Pandas (1.5.0+)**: Data manipulation and analysis
- **NumPy (1.21.0+)**: Numerical computing
- **PyArrow (5.0.0+)**: Efficient data transfer between Pandas and Spark

### Development & Deployment
- **Make**: Build automation and task execution
- **FastAPI (0.95.0+)**: API deployment (planned)
- **Uvicorn (0.20.0+)**: ASGI server
- **Delta Lake (2.0.0+)**: Data lake storage (optional)

### Monitoring & Visualization
- **Matplotlib (3.5.0+)**: Data visualization
- **Seaborn (0.11.0+)**: Statistical visualizations
- **Plotly (5.10.0+)**: Interactive plots
- **W&B (0.15.0+)**: Experiment tracking (optional)

## 🚀 Overview

A production-ready customer churn prediction system built with **PySpark ML** for distributed computing and scalable machine learning. This implementation provides enterprise-grade features including distributed data processing, **Apache Airflow** orchestration for workflow management, real-time inference, and comprehensive MLflow integration for experiment tracking and model registry.

### ✨ Key Features

- 🔥 **Distributed Computing**: Scale from single machine to cluster deployment with Apache Spark
- 🤖 **Advanced ML Pipeline**: End-to-end PySpark ML with 4 algorithms (GBT, Random Forest, Logistic Regression, Decision Tree)
- 📊 **Real-time Streaming**: Live predictions with Structured Streaming
- 🔄 **Airflow Orchestration**: Automated workflow management with 3 production DAGs (ML Pipeline, Hyperparameter Tuning, Model Monitoring)
- 📈 **MLflow Integration**: Experiment tracking, model registry, and artifact management
- 🔧 **Makefile Automation**: Simplified setup and execution with `make` commands
- 🛠️ **Production Ready**: Comprehensive logging, error handling, and monitoring
- 📦 **Modular Architecture**: Separate modules for data ingestion, feature engineering, model training, and inference

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Raw Data       │───▶│  PySpark Data    │───▶│  Processed Data │
│  (CSV/Parquet)  │    │  Pipeline        │    │  (Parquet)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │         Apache Airflow Orchestration          │
        │  ┌──────────────┐  ┌──────────────────────┐  │
        │  │ ML Pipeline  │  │ Hyperparameter Tuning│  │
        │  │     DAG      │  │        DAG           │  │
        │  └──────────────┘  └──────────────────────┘  │
        │  ┌──────────────────────────────────────┐    │
        │  │    Model Monitoring DAG              │    │
        │  └──────────────────────────────────────┘    │
        └───────────────────────────────────────────────┘
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
│  (Real-time)    │    │  Inference       │    │  (JSON Stream)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

> **Note**: Kafka integration is planned for future releases to enable real-time data streaming.

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Required
- Python 3.8+
- Java 8 or 11 (for PySpark)
- Minimum 4GB RAM (8GB+ recommended)

# Optional (for cluster deployment)
- Apache Spark 3.4.0+
- Apache Airflow 2.7.0 (installed via Makefile)
- Hadoop (for YARN deployment)
- Kubernetes (for K8s deployment)
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/vishwajayawickrama/Telco_Churn_Prediction_MLFlow_PySpark_AirFlow_Kafka.git
cd Telco_Churn_Prediction_MLFlow_PySpark_AirFlow_Kafka

# Install dependencies using Makefile
make install

# Activate virtual environment
source .venv/bin/activate
```

### 3. Setup Airflow (Optional for Workflow Orchestration)

```bash
# Setup local Airflow environment
make setup-local-airflow

# Start Airflow services
make airflow-start

# Access Airflow UI at http://localhost:8080
# Username: admin, Password: admin
```

### 4. Run Pipelines

#### Using Makefile (Recommended)

```bash
# Run complete ML pipeline
make run-all

# Or run individual pipelines
make data-pipeline        # Data preprocessing
make train-pipeline       # Model training
make streaming-inference  # Inference pipeline

# Start MLflow UI
make mlflow-ui           # Access at http://localhost:5001
```

#### Using Python Directly

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

### Airflow DAG Orchestration

The project includes three production-ready Airflow DAGs for automated workflow management:

#### 1. ML Pipeline DAG (`telco_churn_ml_pipeline_dag.py`)
Complete end-to-end ML pipeline orchestration:
```python
# Automated workflow includes:
# - Data preprocessing and feature engineering
# - Model training with hyperparameter tuning
# - Model evaluation and validation
# - Inference generation and result storage
# - Slack notifications for pipeline status
```

#### 2. Hyperparameter Tuning DAG (`telco_churn_hyperparameter_tuning_dag.py`)
Automated hyperparameter optimization:
```python
# Parallel tuning for multiple models:
# - Gradient Boosted Trees
# - Random Forest
# - Logistic Regression
# - Decision Tree
# - Best model selection and registration
```

#### 3. Model Monitoring DAG (`telco_churn_model_monitoring_dag.py`)
Production model monitoring and drift detection:
```python
# Continuous monitoring:
# - Model performance tracking
# - Data drift detection
# - Prediction quality monitoring
# - Automated retraining triggers
```

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

### Available Makefile Commands

```bash
# Installation and Setup
make install                 # Install dependencies and create virtual environment
make setup-local-airflow    # Initialize local Airflow environment
make clean                  # Clean up generated artifacts

# Pipeline Execution
make data-pipeline          # Run data preprocessing pipeline
make train-pipeline         # Run model training pipeline
make streaming-inference    # Run streaming inference pipeline
make run-all               # Execute all pipelines sequentially

# Airflow Operations
make airflow-start         # Start Airflow webserver and scheduler
make airflow-stop          # Stop Airflow services
make start-pipeline        # Start complete ML pipeline (Airflow + MLflow)

# MLflow Operations
make mlflow-ui            # Launch MLflow UI (default port: 5001)
make mlflow-clean         # Clean MLflow artifacts
make stop-all             # Stop all MLflow servers

# Development
make help                 # Display all available commands
```

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
  # PySpark ML algorithms supported
  model_type: "gbt_classifier"  # Options: gbt_classifier, random_forest_classifier, 
                                 #          logistic_regression, decision_tree_classifier
  model_types:
    gbt_classifier:              # Gradient Boosted Trees
      max_iter: 100
      max_depth: 10
      step_size: 0.1
      seed: 42
    random_forest_classifier:
      num_trees: 100
      max_depth: 10
      seed: 42
      feature_subset_strategy: "auto"
    logistic_regression:
      max_iter: 1000
      reg_param: 0.01
      elastic_net_param: 0.0
      family: "binomial"
    decision_tree_classifier:
      max_depth: 10
      seed: 42
      impurity: "gini"
```

### Airflow Configuration

Airflow DAGs are configured with:
- **Default Owner**: data-science-team
- **Retries**: 2 with 5-minute delay
- **Execution Timeout**: 4 hours
- **Email Notifications**: Enabled for failures
- **Schedule**: Configurable per DAG (default: daily)

## 🚀 Deployment Options

### Local Development

```bash
# Single machine with all cores (using Makefile)
make run-all

# Or with Airflow orchestration
make start-pipeline
# Access Airflow UI: http://localhost:8080
# Access MLflow UI: http://localhost:5001
```

### Standalone Spark Cluster

```bash
# Start Spark cluster
$SPARK_HOME/sbin/start-master.sh
$SPARK_HOME/sbin/start-workers.sh

# Submit job
spark-submit \
  --master spark://master:7077 \
  --executor-memory 4g \
  --driver-memory 2g \
  pipelines/training_pipeline.py
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

### Run All Pipelines

```bash
# Using Makefile (Recommended)
make run-all

# Individual pipelines
make data-pipeline
make train-pipeline
make streaming-inference
```

### Test Individual Components

```python
# Test data pipeline
from pipelines.data_pipeline import data_pipeline
result = data_pipeline(
    data_path="./data/raw/TelcoCustomerChurnPrediction.csv",
    use_pyspark=True,
    force_rebuild=True
)

# Test model training
from pipelines.training_pipeline import training_pipeline
model = training_pipeline(
    model_type='gbt_classifier',
    use_pyspark=True
)

# Test inference
from pipelines.streaming_inference_pipeline import streaming_inference
prediction = streaming_inference(
    use_pyspark=True,
    input_data=sample_customer
)
```

### Test Airflow DAGs

```bash
# Test DAG syntax
airflow dags list

# Test specific DAG
airflow dags test telco_churn_ml_pipeline_dag 2024-01-01

# Trigger DAG manually
airflow dags trigger telco_churn_ml_pipeline_dag
```

## 📊 Monitoring & Observability

### Airflow UI
- **Local**: http://localhost:8080
- **Credentials**: admin / admin
- Monitor DAG runs, task status, and execution logs
- View task dependencies and execution history

### Spark UI
- **Local**: http://localhost:4040
- **Cluster**: Check cluster manager UI
- Monitor job execution, stages, and tasks
- View executor metrics and storage

### MLflow UI
```bash
# Start MLflow server
make mlflow-ui

# Access at http://localhost:5001
```
- Track experiments and runs
- Compare model performance
- View artifacts and parameters
- Manage model registry

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
.
├── Readme.md                    # Project documentation
├── Makefile                     # Automation commands for setup and execution
├── config.yaml                  # Central configuration file
├── requirements.txt             # Python dependencies
│
├── dags/                        # Apache Airflow DAGs
│   ├── telco_churn_ml_pipeline_dag.py              # Main ML pipeline orchestration
│   ├── telco_churn_hyperparameter_tuning_dag.py    # Hyperparameter optimization
│   └── telco_churn_model_monitoring_dag.py         # Model monitoring and drift detection
│
├── pipelines/                   # End-to-end pipeline implementations
│   ├── data_pipeline.py         # Data ingestion and preprocessing pipeline
│   ├── training_pipeline.py     # Model training and evaluation pipeline
│   └── streaming_inference_pipeline.py  # Real-time inference pipeline
│
├── src/                         # Core PySpark ML modules
│   ├── __init__.py
│   ├── data_ingestion.py        # PySpark data loading and validation
│   ├── handle_missing_values.py # Missing value imputation strategies
│   ├── outlier_detection.py     # Outlier detection and handling
│   ├── feature_binning.py       # Feature binning transformations
│   ├── feature_encoding.py      # PySpark ML encoders (StringIndexer, OneHotEncoder)
│   ├── feature_scaling.py       # Feature scaling (StandardScaler, MinMaxScaler)
│   ├── data_spiltter.py         # Train-test splitting
│   ├── model_building.py        # PySpark ML model builders
│   ├── model_training.py        # Training orchestration with cross-validation
│   ├── model_evaluation.py      # Model evaluation metrics
│   └── model_inference.py       # Batch and streaming inference
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── spark_utils.py           # Spark session management and configuration
│   ├── mlflow_utils.py          # MLflow integration for PySpark
│   ├── airflow_utils.py         # Airflow helper functions
│   ├── config.py                # Configuration management
│   └── logger.py                # Logging configuration
│
└── artifacts/                   # Generated artifacts (gitignored)
    ├── models/                  # Trained PySpark ML pipelines
    ├── data/                    # Processed data (Parquet format)
    ├── evaluation/              # Model evaluation reports
    └── mlflow_training_artifacts/  # MLflow experiment artifacts

Note: data/ directory structure is created at runtime when pipelines are executed.
      Raw data should be placed in data/raw/ directory.
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
- **Apache Airflow** for workflow orchestration and scheduling
- **MLflow** for experiment tracking and model registry
- **PySpark ML** for scalable machine learning capabilities
- **Community** for continuous support and contributions

## 🚧 Future Enhancements

- [ ] **Kafka Integration**: Real-time data streaming from Kafka topics
- [ ] **REST API**: FastAPI-based prediction service
- [ ] **Docker Support**: Containerized deployment
- [ ] **Kubernetes**: Cloud-native orchestration
- [ ] **Advanced Monitoring**: Prometheus and Grafana integration
- [ ] **A/B Testing**: Model comparison in production
- [ ] **Feature Store**: Centralized feature management

---

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: See inline code documentation and config.yaml

---

**🚀 Built for scalable, production-ready machine learning with PySpark, Airflow, and MLflow!**

*Developed with ❤️ for enterprise-scale machine learning and automated workflow orchestration*