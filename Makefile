.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help

# Default Python interpreter
PYTHON = python
VENV = .venv/bin/activate
MLFLOW_PORT ?= 5001

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make setup-local-airflow - Set up local Airflow environment"
	@echo "  make airflow-start       - Start Airflow webserver and scheduler"
	@echo "  make airflow-stop        - Stop Airflow services"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make mlflow-ui           - Start MLflow UI"
	@echo "  make start-pipeline      - Start complete ML pipeline (Airflow + MLflow)"
	@echo "  make clean               - Clean up artifacts"

# Install project dependencies and set up environment
install:
	@echo "Installing project dependencies and setting up environment..."
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	@source .venv/bin/activate && pip install --upgrade pip
	@source .venv/bin/activate && pip install -r requirements.txt
	@source .venv/bin/activate && pip install apache-airflow==2.7.0
	@echo "Installation completed successfully!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Set up local Airflow environment
setup-local-airflow:
	@echo "Setting up local Airflow environment..."
	@source $(VENV) && export AIRFLOW_HOME=~/airflow && airflow db init
	@source $(VENV) && export AIRFLOW_HOME=~/airflow && airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com \
		--password admin
	@echo "Copying DAGs to Airflow directory..."
	@mkdir -p ~/airflow/dags
	@cp -r dags/* ~/airflow/dags/
	@echo "Airflow setup completed!"

# Start Airflow services
airflow-start:
	@echo "Starting Airflow webserver and scheduler..."
	@source $(VENV) && export AIRFLOW_HOME=~/airflow && airflow webserver --port 8080 --daemon
	@source $(VENV) && export AIRFLOW_HOME=~/airflow && airflow scheduler --daemon
	@echo "Airflow services started!"
	@echo "Access Airflow UI at: http://localhost:8080 (admin/admin)"

# Stop Airflow services
airflow-stop:
	@echo "Stopping Airflow services..."
	@-pkill -f "airflow webserver"
	@-pkill -f "airflow scheduler"
	@echo "Airflow services stopped!"

# Start complete ML pipeline
start-pipeline:
	@echo "Starting complete ML pipeline..."
	@make airflow-start
	@sleep 5
	@make mlflow-ui &
	@echo "ML Pipeline started!"
	@echo "Airflow UI: http://localhost:8080"
	@echo "MLflow UI: http://localhost:$(MLFLOW_PORT)"

# Clean up
clean:
	@echo "Cleaning up artifacts..."
	rm -rf artifacts/models/*
	rm -rf artifacts/evaluation/*
	rm -rf artifacts/predictions/*
	rm -rf data/processed/*
	@echo "Cleanup completed!"



# Run data pipeline
data-pipeline:
	@echo "Running data pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py

# Run training pipeline
train-pipeline:
	@echo "Running training pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference:
	@echo "Running streaming inference pipeline with sample JSON..."
	@source $(VENV) && $(PYTHON) pipelines/streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all:
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py
	@echo "\n========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py
	@echo "\n========================================"
	@echo "Step 3: Running streaming inference pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/streaming_inference_pipeline.py
	@echo "\n========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"


mlflow-ui:
	@echo "Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop the server"
	@source $(VENV) && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)

mlflow-clean:
	@echo "Cleaning up MLflow artifacts..."
	rm -rf mlruns
	@echo "MLflow artifacts cleaned!"

# Stop all running MLflow servers
stop-all:
	@echo "Stopping all MLflow servers..."
	@echo "Finding MLflow processes on port $(MLFLOW_PORT)..."
	@-lsof -ti:$(MLFLOW_PORT) | xargs kill -9 2>/dev/null || true
	@echo "Finding other MLflow UI processes..."
	@-ps aux | grep '[m]lflow ui' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@-ps aux | grep '[g]unicorn.*mlflow' | awk '{print $$2}' | xargs kill -9 2>/dev/null || true
	@echo "✅ All MLflow servers have been stopped"