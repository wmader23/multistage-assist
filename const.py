"""Constants for the Multi-Stage Assist integration."""

DOMAIN = "multistage_assist"

# Stage 1: Local Ollama/LLM for Intent Detection
CONF_STAGE1_IP = "stage1_ip"
CONF_STAGE1_PORT = "stage1_port"
CONF_STAGE1_MODEL = "stage1_model"

# Stage 2: Google Gemini for Chat
CONF_GOOGLE_API_KEY = "google_api_key"
CONF_STAGE2_MODEL = "stage2_model" # e.g. "gemini-2.0-flash"