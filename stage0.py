import logging
from typing import Optional, List

from homeassistant.components import conversation
from homeassistant.components.conversation.default_agent import DefaultAgent
from hassil.recognize import recognize_best

from .entity_resolver import EntityResolver

_LOGGER = logging.getLogger(__name__)


class Stage0Result:
    """Container for Stage0 output passed forward."""
    def __init__(self, type_: str, intent=None, raw=None, resolved_ids: Optional[List[str]] = None):
        self.type = type_
        self.intent = intent
        self.raw = raw
        self.resolved_ids = resolved_ids or []


class Stage0Processor:
    """Stage 0: Dry-run NLU and early entity resolution (no LLM)."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.entities = EntityResolver(hass)

    async def _dry_run_recognize(self, user_input: conversation.ConversationInput):
        agent = conversation.async_get_agent(self.hass)
        if not isinstance(agent, DefaultAgent):
            _LOGGER.warning("Only works with DefaultAgent right now")
            return None

        language = user_input.language or "de"
        lang_intents = await agent.async_get_or_load_intents(language)
        if lang_intents is None:
            _LOGGER.debug("No intents loaded for language=%s", language)
            return None

        slot_lists = await agent._make_slot_lists()
        intent_context = agent._make_intent_context(user_input)

        def _run():
            return recognize_best(
                user_input.text,
                lang_intents.intents,
                slot_lists=slot_lists,
                intent_context=intent_context,
                language=language,
                best_metadata_key="hass_custom_sentence",
                best_slot_name="name",
            )

        _LOGGER.debug("Running dry-run recognize for utterance='%s'", user_input.text)
        return await self.hass.async_add_executor_job(_run)

    async def run(self, user_input: conversation.ConversationInput) -> Stage0Result | None:
        result = await self._dry_run_recognize(user_input)
        if not result or not result.intent:
            _LOGGER.debug("NLU did not produce an intent.")
            return None

        entities = {k: v.value for k, v in (result.entities or {}).items()}
        _LOGGER.debug("NLU extracted entities: %s", entities)
        resolved = await self.entities.resolve(entities)
        _LOGGER.debug(
            "Resolved entity_ids: by_area=%s, by_name=%s, merged=%s",
            resolved.by_area, resolved.by_name, resolved.merged
        )

        if not resolved.merged:
            _LOGGER.debug("No entities resolved → Stage1 clarification")
            return Stage0Result("clarification", raw=result)

        threshold = int(getattr(self.config, "early_filter_threshold", 10))
        merged_ids = list(resolved.merged)
        if len(merged_ids) > threshold:
            _LOGGER.debug("Too many entities (%d) → Stage1 clarification", len(merged_ids))
            return Stage0Result("clarification", raw=result)

        return Stage0Result("intent", intent=result.intent, raw=result, resolved_ids=merged_ids)
