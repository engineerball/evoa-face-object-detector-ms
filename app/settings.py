import os

USE_REDIS = os.environ.get("USE_REDIS", False)
ALLOW_LIST_FILE = os.environ.get("ALLOW_LIST_FILE", "data/allow.yaml")
DENY_LIST_FILE = os.environ.get("DENY_LIST_FILE", "data/deny.yaml")