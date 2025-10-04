"""
Documentation and Setup Guide for Telco Customer Churn ML Pipeline with Airflow Integration.

This file provides comprehensive setup instructions and usage documentation
for the complete ML pipeline orchestration system.
"""

# Telco Customer Churn ML Pipeline - Airflow Integration

## 🚀 Overview

This project implements a comprehensive machine learning pipeline for telecommunications customer churn prediction using **PySpark**, **MLflow**, and **Apache Airflow** for orchestration. The pipeline includes data preprocessing, model training, evaluation, inference, hyperparameter tuning, and continuous monitoring.

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Apache Airflow 2.7.0+
- Java 8+ (for PySpark)
- At least 8GB RAM
- 20GB free disk space

### Required Dependencies
```bash
# Core ML Libraries
pyspark==3.4.0
mlflow==2.8.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3

# Airflow (install separately)
# apache-airflow==2.7.0

# Additional Libraries
scipy==1.11.0
joblib==1.3.0
pyyaml==6.0
```

## 🏗️ Architecture Overview

### Pipeline Components
1. **Data Pipeline** (`pipelines/data_pipeline.py`) - PySpark data preprocessing
2. **Training Pipeline** (`pipelines/training_pipeline.py`) - Multi-algorithm model training
3. **Streaming Pipeline** (`pipelines/streaming_inference_pipeline.py`) - Real-time inference
4. **Airflow DAGs** - Workflow orchestration and automation

### Airflow DAGs
1. **Main ML Pipeline** (`telco_churn_ml_pipeline_dag.py`) - Complete ML workflow
2. **Hyperparameter Tuning** (`telco_churn_hyperparameter_tuning_dag.py`) - Model optimization
3. **Model Monitoring** (`telco_churn_model_monitoring_dag.py`) - Performance tracking

## 🛠️ Setup Instructions

### 1. Environment Setup

```bash
# Clone and navigate to project
cd "Telco_Customer_Churn(MLFlow, PySpark, Airflow integrated)"

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements.airflow.txt
```

### 2. Local Airflow Setup

```bash
# Install Airflow
pip install apache-airflow==2.7.0

# Initialize Airflow database
export AIRFLOW_HOME=~/airflow
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Copy DAGs to Airflow directory
cp -r dags/ ~/airflow/dags/

# Start Airflow webserver and scheduler
airflow webserver --port 8080 &
airflow scheduler &
```

### 3. Access Points

After successful setup:
- **Airflow Webserver**: http://localhost:8080 (admin/[your_password])
- **MLflow UI**: Start with `mlflow ui` and access at http://localhost:5000
- **Spark Master UI**: Available when running PySpark applications

## 📊 Data Pipeline Workflow

### 1. Data Preprocessing Pipeline

```python
# Located in: pipelines/data_pipeline.py
# PySpark implementation with 9-step preprocessing:

1. Data Ingestion & Schema Validation
2. Missing Value Handling
3. Data Type Conversions
4. Outlier Detection & Treatment
5. Feature Engineering & Binning
6. Categorical Encoding
7. Feature Scaling
8. Data Splitting (Train/Test)
9. Data Quality Validation
```

### 2. Model Training Pipeline

```python
# Located in: pipelines/training_pipeline.py
# Supports multiple algorithms:

- Gradient Boosted Trees (GBT)
- Random Forest
- Logistic Regression
- Decision Tree

# Features:
- Cross-validation
- Hyperparameter tuning
- MLflow experiment tracking
- Model persistence
```

### 3. Streaming Inference Pipeline

```python
# Located in: pipelines/streaming_inference_pipeline.py
# Capabilities:

- Real-time streaming predictions
- Batch inference
- Model loading from MLflow
- Structured streaming with PySpark
```

## 🚀 Running the Pipeline

### Method 1: Using Airflow (Recommended)

1. **Access Airflow Web UI**:
   ```bash
   # Open browser to http://localhost:8080
   # Login: admin/admin
   ```

2. **Enable and Trigger DAGs**:
   - `telco_churn_ml_pipeline` - Main ML workflow
   - `telco_churn_hyperparameter_tuning` - Model optimization
   - `telco_churn_model_monitoring` - Performance monitoring

3. **Monitor Execution**:
   - View task logs and status
   - Check MLflow experiments
   - Review generated artifacts

### Method 2: Direct Pipeline Execution

```bash
# Run individual pipelines
python pipelines/data_pipeline.py
python pipelines/training_pipeline.py
python pipelines/streaming_inference_pipeline.py
```

## 📈 Airflow DAG Details

### 1. Main ML Pipeline DAG
- **Schedule**: `@daily`
- **Components**:
  - Data quality validation
  - Preprocessing with PySpark
  - Multi-algorithm training
  - Model evaluation
  - Batch inference generation
  - Reporting and notifications

### 2. Hyperparameter Tuning DAG
- **Schedule**: `@weekly`
- **Features**:
  - Grid search for all algorithms
  - Cross-validation
  - MLflow experiment tracking
  - Production config updates

### 3. Model Monitoring DAG
- **Schedule**: `@daily`
- **Monitoring**:
  - Data drift detection
  - Performance degradation alerts
  - Statistical testing (KS, Chi-square)
  - Automated reporting

## 🔧 Configuration

### Main Configuration (`config.yaml`)
```yaml
# Data paths
data:
  raw_path: "data/raw/TelcoCustomerChurnPrediction.csv"
  processed_path: "data/processed/"
  
# Model parameters
model:
  test_size: 0.2
  random_state: 42
  algorithms: ["gbt", "randomforest", "logisticregression"]
  
# MLflow settings
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "telco_churn_prediction"
```

### Airflow Configuration
```bash
# Core settings (in ~/airflow/airflow.cfg)
executor = LocalExecutor
sql_alchemy_conn = sqlite:////Users/[username]/airflow/airflow.db

# Web server
expose_config = True
rbac = False

# Set environment variables
export AIRFLOW_HOME=~/airflow
export PYTHONPATH=$PYTHONPATH:/path/to/your/project
```

## 📊 Monitoring and Alerts

### Performance Thresholds
- **Accuracy**: > 85%
- **F1-Score**: > 80%
- **Precision**: > 78%
- **Recall**: > 82%

### Drift Detection
- **PSI Threshold**: 0.2 (Population Stability Index)
- **KS Threshold**: 0.3 (Kolmogorov-Smirnov)
- **Chi-square**: p-value < 0.05

### Alert Channels
- Email notifications
- Slack integration (configurable)
- Airflow task failure alerts

## 🔄 Model Lifecycle

### 1. Development Phase
```bash
# Data exploration and preprocessing
python pipelines/data_pipeline.py

# Model experimentation
python pipelines/training_pipeline.py
```

### 2. Training Phase
```bash
# Trigger hyperparameter tuning DAG
# Airflow UI: telco_churn_hyperparameter_tuning

# Review MLflow experiments
# Browser: http://localhost:5000
```

### 3. Production Phase
```bash
# Deploy main pipeline DAG
# Schedule: @daily automatic execution

# Monitor performance
# Airflow UI: telco_churn_model_monitoring
```

### 4. Maintenance Phase
```bash
# Review monitoring reports
# Check drift detection alerts
# Retrain models when needed
```

## 📁 Project Structure

```
Telco_Customer_Churn(MLFlow, PySpark, Airflow integrated)/
├── config.yaml                           # Main configuration
├── requirements.txt                       # Python dependencies
├── requirements.txt                      # Python dependencies
├── 
├── data/
│   ├── raw/                              # Original datasets
│   └── processed/                        # Processed datasets
├── 
├── pipelines/
│   ├── data_pipeline.py                  # PySpark preprocessing
│   ├── training_pipeline.py              # Model training
│   └── streaming_inference_pipeline.py   # Real-time inference
├── 
├── dags/
│   ├── telco_churn_ml_pipeline_dag.py    # Main ML workflow
│   ├── telco_churn_hyperparameter_tuning_dag.py  # Optimization
│   ├── telco_churn_model_monitoring_dag.py       # Monitoring
│   └── airflow_utils.py                  # Utility functions
├── 
├── src/
│   ├── data_ingestion.py                 # Data loading
│   ├── feature_engineering.py            # Feature processing
│   ├── model_building.py                 # Model creation
│   └── model_evaluation.py               # Performance metrics
├── 
├── artifacts/
│   ├── models/                           # Trained models
│   ├── encoders/                         # Feature encoders
│   └── data/                             # Processed datasets
└── 
└── utils/
    ├── config.py                         # Configuration management
    └── logger.py                         # Logging utilities
```

## 🚨 Troubleshooting

### Common Issues

1. **Airflow Connection Issues**
   ```bash
   # Reset Airflow database
   airflow db reset
   airflow db init
   ```

2. **PySpark Memory Issues**
   ```python
   # Increase driver memory in spark configuration
   spark.conf.set("spark.driver.memory", "4g")
   spark.conf.set("spark.executor.memory", "4g")
   ```

3. **MLflow Connection Issues**
   ```bash
   # Start MLflow tracking server
   mlflow server --host 127.0.0.1 --port 5000
   ```

### Logs and Debugging
- **Airflow Logs**: Available in Web UI task instances and `~/airflow/logs/`
- **Application Logs**: `logs/` directory
- **MLflow Logs**: Console output when running `mlflow ui`
- **Spark Logs**: Available in Spark application UI and local logs

## 📝 Best Practices

### 1. Data Management
- Regular data quality checks
- Version control for datasets
- Proper data lineage tracking

### 2. Model Management
- Experiment tracking with MLflow
- Model versioning and registry
- A/B testing for model deployment

### 3. Pipeline Management
- Comprehensive error handling
- Resource optimization
- Monitoring and alerting

### 4. Security
- Secure credential management
- Access control configuration
- Data encryption in transit

## 🔮 Future Enhancements

### Planned Features
1. **Advanced Monitoring**
   - Feature importance drift
   - Model explainability tracking
   - Business metric correlation

2. **Auto-ML Integration**
   - Automated feature selection
   - Neural architecture search
   - AutoML pipeline optimization

3. **Production Deployment**
   - Kubernetes orchestration
   - Model serving with MLflow
   - Blue-green deployments

4. **Enhanced Streaming**
   - Kafka integration
   - Real-time feature stores
   - Stream processing optimization

## 📞 Support

For questions, issues, or contributions:
- Create GitHub issues for bugs
- Submit pull requests for improvements
- Contact the ML Engineering team

---

**Last Updated**: 2024  
**Version**: 2.0.0  
**Maintainer**: Data Science Team