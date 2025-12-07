import os
# Use os.getenv to ensure the subprocess picks up UI settings
LOCAL_LLM_API_URL = os.getenv("LOCAL_LLM_API_URL", "http://10.2.0.2:1234")
LOCAL_MAIN_MODEL = os.getenv("LOCAL_MAIN_MODEL", "local-model")