import os


global ENABLE_PROFILING_DEBUG
ENABLE_PROFILING_DEBUG = os.getenv("ENABLE_PROFILING_DEBUG", "false").lower() == "true"

global ENABLE_GRAPH_MODE
ENABLE_GRAPH_MODE = os.getenv("ENABLE_GRAPH_MODE", "false").lower() == "true"
