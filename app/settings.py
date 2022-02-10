import os

USE_REDIS = os.environ.get("USE_REDIS", False)
ALLOW_LIST_FILE = os.environ.get("ALLOW_LIST_FILE", "data/allow.yaml")
DENY_LIST_FILE = os.environ.get("DENY_LIST_FILE", "data/deny.yaml")
TENSORFLOW_URL = os.env.get("TENSORFLOW_URL", "http://tensorflow:8501/v1/models/rfcn:predict")