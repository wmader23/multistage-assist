"""Conversation agent for Multi-Stage Assist."""
from __future__ import annotations

import logging
import os
import re

import spacy

from homeassistant.helpers import intent
from homeassistant.components import conversation
from hassil.recognize import recognize_best

_LOGGER = logging.getLogger(__name__)

# Base directory & path to spaCy model
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "de_core_news_sm", "de_core_news_sm-3.7.0")


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Multi-Stage Assist Agent for Home Assistant."""

    def __init__(self, hass, config):
        """Initialize the agent."""
        self.hass = hass
        self.config = config
        self._nlp = None
        try:
            self._nlp = spacy.load(MODEL_DIR)
        except Exception as e:
            _LOGGER.error("Error loading spaCy model: %s", e)

    @property
    def supported_languages(self) -> set[str]:
        """Return the languages supported by this agent."""
        return {"de"}

    async def _dry_run_recognize(
        self, utterance: str, language: str, user_input: conversation.ConversationInput
    ):
        """Perform dry-run intent recognition using HA's DefaultAgent."""
        agent = conversation.async_get_agent(self.hass)
        if not isinstance(agent, conversation.DefaultAgent):
            _LOGGER.warning(
                "MultiStageAssistAgent only works with DefaultAgent right now"
            )
            return None

        lang_intents = await agent.async_get_or_load_intents(language)
        if lang_intents is None:
            return None

        slot_lists = await agent._make_slot_lists()
        intent_context = agent._make_intent_context(user_input)

        def _run_recognizer():
            return recognize_best(
                utterance,
                lang_intents.intents,
                slot_lists=slot_lists,
                intent_context=intent_context,
                language=language,
                best_metadata_key="hass_custom_sentence",
                best_slot_name="name",
            )

        # Offload blocking I/O to executor
        return await self.hass.async_add_executor_job(_run_recognizer)

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process user input with multi-stage routing, probing HA NLU first."""
        utterance = user_input.text
        language = user_input.language or "de"
        _LOGGER.info("Received utterance: %s", utterance)

        try:
            # ---- STEP 1: Dry-run recognition ----
            result = await self._dry_run_recognize(utterance, language, user_input)
            if not result:
                _LOGGER.info("Dry-run did not recognize -> Stage 2 fallback")
                return await self._call_stage2_llm(user_input)

            _LOGGER.debug("Dry-run RecognizeResult: %s", result)

            entities = result.entities or {}
            _LOGGER.debug("NLU extracted entities: %s", entities)

            # ---- STEP 2: Constraint checks ----
            if len(entities) > 1:
                allow_multiple = False
                if self._nlp:
                    utterance_lower = utterance.lower()
                    doc = self._nlp(utterance_lower)
                    if re.search(r"\balle(n|r|s)?\b", utterance_lower):
                        allow_multiple = True
                    else:
                        for token in doc:
                            if token.pos_ in {"NOUN", "PROPN"} and token.morph.get(
                                "Number"
                            ) == ["Plur"]:
                                allow_multiple = True
                                break

                if not allow_multiple:
                    _LOGGER.info("Ambiguity detected -> Stage 1 clarification")
                    return await self._call_stage1_clarification(user_input, result)

            # ---- STEP 3: Confirmation / Execution ----
            if self.config.get("require_confirmation", True):
                confirm_msg = f"Soll ich das wirklich ausführen: '{utterance}'?"
                return await self._make_placeholder_result(
                    confirm_msg, language=language
                )

            return await conversation.async_converse(
                self.hass,
                text=utterance,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=language,
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.warning("NLU probe failed: %s", err)
            return await self._call_stage2_llm(user_input)

    async def _make_placeholder_result(
        self, message: str, language: str | None = None
    ) -> conversation.ConversationResult:
        lang = language or "de"
        resp = intent.IntentResponse(language=lang)
        resp.response_type = intent.IntentResponseType.QUERY_ANSWER
        resp.async_set_speech(message)
        return conversation.ConversationResult(response=resp)

    async def _call_stage1_classification(self, user_input, resp=None):
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
