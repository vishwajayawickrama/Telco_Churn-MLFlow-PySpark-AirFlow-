# Telco Customer Churn Prediction

A comprehensive machine learning project to predict customer churn for a telecommunications company using advanced data pipelines and multiple ML algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Data Pipeline](#data-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Contributing](#contributing)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline to predict customer churn in the telecommunications industry. The pipeline includes comprehensive data preprocessing, feature engineering, model training, evaluation, and inference capabilities.

**Key Objectives:**
- Predict which customers are likely to churn
- Identify key factors contributing to customer churn
- Provide actionable insights for customer retention strategies
- Implement scalable ML pipelines for production deployment

## 📁 Project Structure

```
├── config.yaml                     # Configuration file for all pipeline parameters
├── Makefile                        # Automation commands for easy execution
├── requirements.txt                # Python dependencies
├── Readme.md                       # Project documentation
├── artifacts/                      # Generated artifacts
│   ├── data/                       # Processed datasets (train/test splits)
│   └── encode/                     # Trained encoders for categorical features
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Processed datasets at different stages
├── pipelines/                      # ML pipelines
│   ├── data_pipeline.py           # Data preprocessing pipeline
│   ├── training_pipeline.py       # Model training pipeline
│   └── streaming_inference_pipeline.py # Real-time inference pipeline
├── src/                           # Core source code modules
│   ├── data_ingestion.py          # Data loading and validation
│   ├── handle_missing_values.py   # Missing value handling strategies
│   ├── outlier_detection.py       # Outlier detection and treatment
│   ├── feature_binning.py         # Feature binning strategies
│   ├── feature_encoding.py        # Categorical encoding techniques
│   ├── feature_scaling.py         # Feature scaling methods
│   ├── data_spiltter.py          # Train/test splitting
│   ├── model_building.py          # Model factory and configuration
│   ├── model_training.py          # Model training logic
│   ├── model_evaluation.py        # Model evaluation metrics
│   └── model_inference.py         # Prediction and inference
└── utils/
    └── config.py                  # Configuration management utilities
```

## ✨ Features

### Data Processing
- **Automated Data Ingestion**: CSV data loading with validation
- **Missing Value Handling**: Multiple strategies (drop, fill, impute)
- **Outlier Detection**: IQR-based outlier detection and treatment
- **Feature Engineering**: Custom binning and transformation strategies
- **Encoding**: Support for both nominal and ordinal categorical variables
- **Scaling**: Min-Max scaling for numerical features
- **Train/Test Splitting**: Stratified splitting with configurable ratios

### Machine Learning
- **Multiple Algorithms**: Support for various ML algorithms (XGBoost, LightGBM, etc.)
- **Model Factory Pattern**: Easy model instantiation and configuration
- **Comprehensive Evaluation**: Multiple metrics for model assessment
- **Cross-Validation**: Robust model validation techniques
- **Hyperparameter Tuning**: Automated parameter optimization

### Production Ready
- **Streaming Inference**: Real-time prediction capabilities
- **Configuration Management**: YAML-based configuration system
- **Artifact Management**: Systematic storage of models and encoders
- **Pipeline Automation**: Makefile commands for easy execution

## 🔄 Data Pipeline

The data processing pipeline consists of the following steps:

### 1. **Data Ingestion**
   - Load raw customer data from CSV files
   - Validate data schema and format
   - Handle data type conversions

### 2. **Missing Value Handling**
   - Identify missing values across all features
   - Apply appropriate strategies based on feature type
   - Document missing value patterns

### 3. **Outlier Detection & Treatment**
   - Use IQR method to identify outliers
   - Apply treatment strategies (removal, capping, transformation)
   - Preserve data integrity while handling extremes

### 4. **Feature Engineering**
   - **Binning**: Convert continuous variables to categorical bins
   - **Encoding**: Transform categorical variables to numerical format
     - Nominal encoding for unordered categories
     - Ordinal encoding for ordered categories
   - **Scaling**: Normalize numerical features using Min-Max scaling

### 5. **Data Splitting**
   - Stratified train-test split to maintain class distribution
   - Configurable split ratios
   - Separate feature and target variables

### 6. **Artifact Generation**
   - Save processed datasets
   - Store trained encoders for inference
   - Generate data quality reports

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Mini Project 01"
   ```

2. **Install dependencies and setup environment**
   ```bash
   make install
   ```
   This command will:
   - Create a virtual environment
   - Install all required packages
   - Setup the project structure

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

## 💻 Usage

### Quick Start
Run all pipelines in sequence:
```bash
make run-all
```

### Individual Pipeline Execution

1. **Data Pipeline** (Preprocessing)
   ```bash
   make data-pipeline
   ```

2. **Training Pipeline** (Model Training)
   ```bash
   make train-pipeline
   ```

3. **Streaming Inference** (Real-time Predictions)
   ```bash
   make streaming-inference
   ```

### Manual Execution

You can also run pipelines directly with Python:
```bash
# Data preprocessing
python pipelines/data_pipeline.py

# Model training
python pipelines/training_pipeline.py

# Streaming inference
python pipelines/streaming_inference_pipeline.py
```

### Cleanup
Remove generated artifacts:
```bash
make clean
```

## 📊 Model Performance

The project supports multiple machine learning algorithms:

- **XGBoost**: Gradient boosting for high performance
- **LightGBM**: Fast gradient boosting with low memory usage
- **Random Forest**: Ensemble method for robust predictions
- **Logistic Regression**: Linear baseline model

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed prediction breakdown

## ⚙️ Configuration

The project uses a YAML configuration file (`config.yaml`) for easy customization:

### Key Configuration Sections

**Data Paths**
```yaml
data_paths:
  raw_data: "../data/raw/TelcoCustomerChurnPrediction.csv"
  processed_data: "../data/processed/..."
  artifacts_dir: "artifacts"
```

**Feature Configuration**
```yaml
columns:
  target: "Churn"
  drop_columns: ["customerID"]
  nominal_columns: ['gender', 'Partner', 'PhoneService', ...]
  ordinal_columns: ['Contract']
  numeric_columns: ['MonthlyCharges', 'TotalCharges', 'tenure']
```

**Model Parameters**
- Training parameters
- Hyperparameter ranges
- Evaluation metrics
- Cross-validation settings

## 🛠️ Development

### Adding New Features
1. Implement new preprocessing steps in the `src/` directory
2. Update the data pipeline to include new steps
3. Modify configuration file as needed
4. Test the changes thoroughly

### Adding New Models
1. Extend the `ModelFactory` class in `model_building.py`
2. Add model-specific parameters to the configuration
3. Update the training pipeline
4. Validate performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📋 Requirements

See `requirements.txt` for a complete list of dependencies. Key libraries include:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **xgboost**: Gradient boosting framework
- **lightgbm**: Gradient boosting framework
- **matplotlib/seaborn**: Data visualization
- **pyyaml**: Configuration management

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Business Impact

This churn prediction model enables:
- **Proactive Customer Retention**: Identify at-risk customers before they churn
- **Targeted Marketing**: Focus retention efforts on high-value customers
- **Resource Optimization**: Allocate retention budgets more effectively
- **Revenue Protection**: Reduce customer acquisition costs by retaining existing customers

---

**Note**: This project is designed for educational and demonstration purposes. For production deployment, additional considerations such as data privacy, model monitoring, and A/B testing should be implemented.