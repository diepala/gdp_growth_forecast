"""
Config module.

This module contains some constants and global variables used in the project
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_PATH = os.path.join(BASE_DIR, "db.sqlite3")

MODELS_PATH = os.path.join(BASE_DIR, "models")

MODEL_FILE_PATH = os.path.join(MODELS_PATH, 'trained_model.model')

# path where the selected feature names are stored
FEATURES_PATH = os.path.join(BASE_DIR, "features", "selected_features.json")

LOGS_PATH = os.path.join(BASE_DIR, "logs")

LOG_CONFIG_PATH = os.path.join(BASE_DIR, 'log.ini')

# target indicator code. The indicator to predict
TARGET = 'NY.GDP.MKTP.KD.ZG'