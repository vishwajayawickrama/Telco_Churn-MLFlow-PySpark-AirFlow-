import os
import sys
import json
import pandas as pd

# Add src and utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import modules
from model_inference import ModelInference
from config import get_model_config, get_inference_config
from logger import get_logger, ProjectLogger, log_exceptions

# Initialize logger
logger = get_logger(__name__)