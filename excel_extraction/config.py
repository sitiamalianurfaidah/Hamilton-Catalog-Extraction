"""
Configuration file for Catalog Extractor
"""

import os
from pathlib import Path

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "10.5.0.4")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "hamiltonserver3.14")

DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Schema and Table
SCHEMA_NAME = "construction"
TABLE_NAME = "catalog_items_hf"

# Qwen API Configuration
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-0fc24c229d174d1b99624a49544b1c7a")
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MAX_MODEL = "qwen-max"

# Paths
BASE_DIR = Path(__file__).parent
DATA_OUTPUT_DIR = BASE_DIR / "Exel"
DATA_OUTPUT_DIR.mkdir(exist_ok=True)

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384