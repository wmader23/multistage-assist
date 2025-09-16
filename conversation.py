"""Conversation agent for Multi-Stage Assist."""
from __future__ import annotations

import json
import logging
import os
import re
import spacy

from typing import Set
from homeassistant.helpers import intent
from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AbstractConversationAgent,
    ConversationResult,
    HOME_ASSISTANT_AGENT,
    async_handle_intents,
)

_LOGGER = logging.getLogger(__name__)

# Base directory & path to spaCy model
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "de_core_news_sm", "de_core_news_sm-3.7.0")


class MultiStageAssistAgent(AbstractConversationAgent):
    """Multi-Stage Assist Agent for Home Assistant."""

    def __init__(self, hass, config):
        """Initialize the agent."""
        self.hass = hass
        self.config = config
        try:
            self._nlp = spacy.load(MODEL_DIR)
        except Exception as e:
            _LOGGER.error("Error loading spaCy model: %s", e)
            self._nlp = None

    @property
    def supported_languages(self) -> Set[str]:
        """Return the languages supported by this agent."""
        return {"de"}

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process user input with multi-stage routing, probing HA NLU first."""
        utterance = user_input.text
        _LOGGER.info("Received utterance: %s", utterance)

        try:
            # ---- STEP 1: Probe built-in NLU (parse only, no execution) ----
            resp = await async_handle_intents(self.hass, user_input)

            # Log full result
            if resp:
                try:
                    _LOGGER.debug(
                        "Full NLU result: %s",
                        json.dumps(resp.as_dict(), indent=2, default=str),
                    )
                except Exception:
                    _LOGGER.debug("Full NLU result (raw dict): %s", resp.as_dict())

            if resp is None:
                _LOGGER.info("Built-in NLU returned no result -> fallback Stage 2")
                return await self._call_stage2_llm(user_input)

            # Extract targets
            try:
                data = resp.as_dict().get("data") or {}
            except Exception:
                data = {}
            targets = data.get("targets") or []
            _LOGGER.debug("NLU extracted targets: %s", targets)

            # ---- STEP 2: Constraint checks ----
            if targets and len(targets) > 1 and self._nlp:
                utterance_lower = utterance.lower()
                doc = self._nlp(utterance_lower)

                allow_multiple = False

                if re.search(r"\balle(n|r|s)?\b", utterance_lower):
                    allow_multiple = True

                if not allow_multiple:
                    for token in doc:
                        if token.pos_ in {"NOUN", "PROPN"} and token.morph.get("Number") == ["Plur"]:
                            _LOGGER.debug(
                                "Detected plural noun: %s (lemma=%s, morph=%s)",
                                token.text,
                                token.lemma_,
                                token.morph,
                            )
                            allow_multiple = True
                            break

                if not allow_multiple:
                    _LOGGER.info(
                        "Ambiguity detected: multiple targets %s without 'alle' or plural -> clarification",
                        targets,
                    )
                    return await self._call_stage1_clarification(user_input, resp)

            # Preview speech
            speech_preview = None
            if isinstance(getattr(resp, "speech", None), dict):
                speech_preview = (
                    (resp.speech.get("plain") or resp.speech.get("ssml") or {})
                ).get("speech")

            _LOGGER.info(
                "Built-in NLU parsed response_type=%s error_code=%s speech=%s",
                getattr(resp, "response_type", None),
                getattr(resp, "error_code", None),
                speech_preview,
            )

            # ---- STEP 3: Handle errors ----
            if resp.response_type == intent.IntentResponseType.ERROR:
                if resp.error_code == intent.IntentResponseErrorCode.NO_INTENT_MATCH:
                    return await self._call_stage1_classification(user_input)
                if resp.error_code == intent.IntentResponseErrorCode.NO_VALID_TARGETS:
                    return await self._call_stage1_clarification(user_input, resp)
                if resp.error_code == intent.IntentResponseErrorCode.FAILED_TO_HANDLE:
                    return await self._call_stage2_llm(user_input, resp)
                return await self._call_stage2_llm(user_input, resp)

            _LOGGER.info("Confirmed -> sending to conversation.async_converse for execution")
            return await conversation.async_converse(
                self.hass,
                text=utterance,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=user_input.language,
                agent_id=HOME_ASSISTANT_AGENT,  # make sure built-in executes
            )

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.warning("Built-in NLU probe failed: %s", err)

        return await self._call_stage2_llm(user_input)

    async def _make_placeholder_result(
        self, message: str, language: str | None = None
    ) -> ConversationResult:
        lang = language or "de"
        resp = intent.IntentResponse(language=lang)
        resp.response_type = intent.IntentResponseType.QUERY_ANSWER
        resp.async_set_speech(message)
        return ConversationResult(response=resp)

    async def _call_stage1_classification(self, user_input):
        msg = "Stage 1 classification called"
        _LOGGER.debug(msg)
        return await self._make_placeholder_result(msg, language=user_input.language)

    async def _call_stage1_clarification(self, user_input, resp):
        msg = "Stage 1 clarification called"
        _LOGGER.debug(msg)
        return await self._make_placeholder_result(msg, language=user_input.language)

    async def _call_stage2_llm(self, user_input, resp=None):
        msg = "Stage 2 LLM called"
        _LOGGER.debug(msg)
        return await self._make_placeholder_result(msg, language=user_input.language)
