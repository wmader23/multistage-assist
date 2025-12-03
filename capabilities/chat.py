import logging
import asyncio
from typing import Any, Dict, List

from homeassistant.components import conversation
from custom_components.multistage_assist.const import CONF_GOOGLE_API_KEY, CONF_STAGE2_MODEL
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class ChatCapability(Capability):
    """
    Handle general conversation using Google Gemini (via google-genai SDK).
    Initializes the client lazily to avoid blocking I/O in the event loop.
    """

    name = "chat"
    description = "General conversation handler using Google Gemini."

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.api_key = config.get(CONF_GOOGLE_API_KEY)
        self.model_name = config.get(CONF_STAGE2_MODEL, "gemini-1.5-flash")
        
        self._client_wrapper = None
        self._init_lock = asyncio.Lock()

    async def _get_client(self):
        """
        Initialize the GoogleGeminiClient in an executor to avoid blocking the loop
        during SSL certificate loading AND module imports.
        """
        async with self._init_lock:
            if self._client_wrapper:
                return self._client_wrapper
            
            if not self.api_key:
                return None

            # Define the blocking creation function
            def _create_client():
                # Perform the heavy import inside the thread, not at module level
                # This prevents the 'import google.genai' from blocking the main loop
                from .google_gemini_client import GoogleGeminiClient
                return GoogleGeminiClient(api_key=self.api_key, model=self.model_name)

            try:
                _LOGGER.debug("[Chat] Initializing Gemini Client (and importing SDK) in background thread...")
                self._client_wrapper = await self.hass.async_add_executor_job(_create_client)
                _LOGGER.debug("[Chat] Gemini Client initialized successfully.")
            except Exception as e:
                _LOGGER.error("Failed to initialize Google GenAI client: %s", e)
                
            return self._client_wrapper

    async def run(self, user_input, history: List[Dict[str, str]] = None, **_: Any) -> conversation.ConversationResult:
        # Get the client (initializing it if necessary)
        client = await self._get_client()

        if not client:
            _LOGGER.error("[Chat] No Google API key configured or client initialization failed.")
            response_text = "Ich bin nicht für Chat konfiguriert (Client Fehler)."
        else:
            _LOGGER.debug("[Chat] Sending to Gemini (model=%s) with %d history items.", client.model, len(history) if history else 0)
            response_text = await client.chat(user_input.text, history)

        intent_response = conversation.intent.IntentResponse(language=user_input.language or "de")
        intent_response.async_set_speech(response_text)
        
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )
