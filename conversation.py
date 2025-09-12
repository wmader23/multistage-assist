"""Conversation agent for Multi-Stage Assist."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.helpers import intent

_LOGGER = logging.getLogger(__name__)


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Multi-Stage Assist router agent."""

    def __init__(self, hass, config: dict[str, Any]) -> None:
        self.hass = hass
        self.config = config

    @property
    def attribution(self) -> dict[str, Any]:
        return {"name": "Multi-Stage Assist"}

    @property
    def supported_languages(self) -> list[str] | str:
        return conversation.MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process user input with multi-stage routing."""
        utterance = user_input.text
        _LOGGER.info("Received utterance: %s", utterance)

        # --------------------------------------
        # Zero stage: Forward to HA’s built-in NLU
        # --------------------------------------
        try:
            result = await conversation.async_converse(
                self.hass,
                text=utterance,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=user_input.language,
            )
            if result:
                resp = result.response
                _LOGGER.info(
                    "HA NLU returned response_type=%s error_code=%s speech=%s",
                    getattr(resp, "response_type", None),
                    getattr(resp, "error_code", None),
                    getattr(getattr(resp, "speech", None), "plain", None),
                )

                # Successful action / answer
                if resp.response_type != intent.IntentResponseType.ERROR:
                    _LOGGER.info("HA NLU handled utterance successfully")
                    return result

                # --------------------------------------
                # Handle ERROR responses explicitly
                # --------------------------------------
                if resp.error_code == intent.IntentResponseErrorCode.NO_INTENT_MATCH:
                    _LOGGER.info("NLU: No intent match -> Stage 1 classification")
                    return await self._call_stage1_classification(user_input)

                if resp.error_code == intent.IntentResponseErrorCode.NO_VALID_TARGETS:
                    _LOGGER.info("NLU: No valid targets -> Stage 1 clarification")
                    return await self._call_stage1_clarification(user_input, resp)

                if resp.error_code == intent.IntentResponseErrorCode.FAILED_TO_HANDLE:
                    _LOGGER.info("NLU: Failed to handle -> escalate to Stage 2")
                    return await self._call_stage2_llm(user_input, resp)

                _LOGGER.info(
                    "NLU returned error not explicitly handled (%s). Falling back to Stage 2.",
                    resp.error_code,
                )
                return await self._call_stage2_llm(user_input, resp)

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.warning("HA NLU failed: %s", err)

        # If NLU completely fails (exception), go to Stage 2
        return await self._call_stage2_llm(user_input)

    # ------------------------------------------------------------------
    # Stage 1: Classification (no intent match)
    # ------------------------------------------------------------------
    async def _call_stage1_classification(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        system_prompt = (
            "What is the intention of the given prompt? "
            "Can you identify the area of interest and the possible device? "
            "If unclear or ambiguous return ERROR."
        )
        _LOGGER.info("Stage 1 classification system prompt: %s", system_prompt)
        # TODO: Call Ollama API here
        ir = intent.IntentResponse(language=user_input.language)
        ir.async_set_speech("(Stage1 Classification -> would call Ollama here)")
        return conversation.ConversationResult(response=ir, conversation_id=user_input.conversation_id)

    # ------------------------------------------------------------------
    # Stage 1: Clarification (multiple devices)
    # ------------------------------------------------------------------
    async def _call_stage1_clarification(
        self,
        user_input: conversation.ConversationInput,
        resp: intent.IntentResponse,
    ) -> conversation.ConversationResult:
        system_prompt = (
            "Form only a single clarification question. If multiple items are provided, "
            "respond that there are multiple items and ask the user to choose which one they mean. "
            "Format the options in a natural language list (with commas and or before the last option).\n\n"
            "Example\n"
            "Input: ['light.office','light.kitchen','light.hallway']\n"
            "Output: I found more than one light. Please choose the one you want: Office, Kitchen, or Hallway?"
        )
        _LOGGER.info("Stage 1 clarification system prompt: %s", system_prompt)
        # TODO: Pass list of devices from resp to Ollama
        ir = intent.IntentResponse(language=user_input.language)
        ir.async_set_speech("(Stage1 Clarification -> would call Ollama here)")
        return conversation.ConversationResult(response=ir, conversation_id=user_input.conversation_id)

    # ------------------------------------------------------------------
    # Stage 2: Fallback LLM
    # ------------------------------------------------------------------
    async def _call_stage2_llm(
        self,
        user_input: conversation.ConversationInput,
        resp: intent.IntentResponse | None = None,
    ) -> conversation.ConversationResult:
        _LOGGER.info("Stage 2 fallback LLM invoked (final catch-all)")
        # TODO: Call Stage 2 LLM API here
        ir = intent.IntentResponse(language=user_input.language)
        ir.async_set_speech("(Stage2 -> would call LLM here)")
        return conversation.ConversationResult(response=ir, conversation_id=user_input.conversation_id)
