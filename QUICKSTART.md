# Telco Customer Churn ML Pipeline - Quick Start

## 🚀 One-Command Setup

```bash
# Start the complete ML pipeline
make start-pipeline
```

## 📋 Quick Access

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow Webserver | http://localhost:8080 | admin/admin |
| MLflow UI | http://localhost:5000 | - |
| Spark Master | http://localhost:8081 | - |

## 🔄 Available Commands

```bash
# Setup and Installation
make install              # Install all dependencies
make setup-airflow        # Initialize Airflow
make start-services       # Start Docker services

# Pipeline Execution
make run-preprocessing    # Run data preprocessing
make run-training         # Run model training
make run-inference        # Run batch inference
make run-full-pipeline    # Execute complete pipeline

# Airflow Operations
make airflow-start        # Start Airflow services
make airflow-stop         # Stop Airflow services
make airflow-restart      # Restart Airflow
make airflow-logs         # View Airflow logs

# Development
make test                 # Run all tests
make lint                 # Code linting
make format               # Format code
make clean                # Clean artifacts

# Monitoring
make check-status         # Check all services
make view-logs            # View application logs
make monitor-models       # Model performance monitoring
```

## 📊 DAG Execution Order

1. **telco_churn_ml_pipeline** (Daily)
   - Main ML workflow
   - Auto-triggered daily at midnight

2. **telco_churn_hyperparameter_tuning** (Weekly)
   - Model optimization
   - Runs every Sunday

3. **telco_churn_model_monitoring** (Daily)
   - Performance monitoring
   - Drift detection

## 🛠️ Development Workflow

```bash
# 1. Setup environment
make install
make setup-airflow

# 2. Start services
make start-services

# 3. Run pipeline
make run-full-pipeline

# 4. Monitor results
# Open http://localhost:8080 in browser
# View experiments at http://localhost:5000
```

## 🚨 Troubleshooting

```bash
# Reset everything
make clean-all
make setup-airflow
make start-services

# Check service status
make check-status

# View logs
make view-logs
```

---
**Quick tip**: Run `make help` to see all available commands!