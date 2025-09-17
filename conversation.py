"""Conversation agent for Multi-Stage Assist."""
from __future__ import annotations

import logging
import json
import dataclasses
from homeassistant.helpers import intent, area_registry, entity_registry
from homeassistant.components import conversation
from hassil.recognize import recognize_best

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ResolvedEntities:
    by_area: list[str]
    by_name: list[str]

    @property
    def merged(self) -> list[str]:
        return list({*self.by_area, *self.by_name})


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Multi-Stage Assist Agent for Home Assistant."""

    def __init__(self, hass, config):
        """Initialize the agent."""
        self.hass = hass
        self.config = config

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

        return await self.hass.async_add_executor_job(_run_recognizer)

    async def _resolve_entities(self, entities: dict[str, str]) -> ResolvedEntities:
        """Resolve slots into entity_ids (clustered by area vs. name)."""
        ent_reg = entity_registry.async_get(self.hass)
        area_reg = area_registry.async_get(self.hass)

        domain = entities.get("domain")
        area_name = entities.get("area")
        name = entities.get("name")

        by_area: list[str] = []
        by_name: list[str] = []

        for ent in ent_reg.entities.values():
            if domain and ent.domain != domain:
                continue

            # Match by area
            if area_name:
                area_id = ent.area_id
                if not area_id:
                    continue
                area = area_reg.async_get_area(area_id)
                if area and area.name.lower() == area_name.lower():
                    by_area.append(ent.entity_id)

            # Match by name
            if name and ent.original_name and name.lower() in ent.original_name.lower():
                by_name.append(ent.entity_id)

        # ---- fallback: match area_name against entity name ----
        if area_name and not by_area:
            for ent in ent_reg.entities.values():
                if domain and ent.domain != domain:
                    continue
                if ent.original_name and area_name.lower() in ent.original_name.lower():
                    by_name.append(ent.entity_id)

        return ResolvedEntities(by_area=by_area, by_name=by_name)

    async def _execute_intent(
        self,
        intent_name: str,
        entities: dict[str, str],
        language: str,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Execute HA intent via async_handle with rewritten slots."""
        slots = {k: {"value": v} for k, v in entities.items()}

        resp = await intent.async_handle(
            self.hass,
            platform="conversation",
            intent_type=intent_name,
            slots=slots,
            text_input=user_input.text,
            context=user_input.context,
            language=language,
            conversation_agent_id="multistage_assist",
        )

        return conversation.ConversationResult(response=resp)

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

            # ---- Debug logging ----
            try:
                result_dict = {
                    "intent": result.intent.name if result.intent else None,
                    "response": result.response,
                    "entities": {k: v.value for k, v in (result.entities or {}).items()},
                }
                _LOGGER.debug("Dry-run result: %s", json.dumps(result_dict, indent=2))
            except Exception as err:
                _LOGGER.debug("Could not log RecognizeResult: %s", err)

            entities = {k: v.value for k, v in (result.entities or {}).items()}
            _LOGGER.debug("NLU extracted entities: %s", entities)

            resolved = await self._resolve_entities(entities)
            _LOGGER.debug(
                "Dry-run resolved entity_ids (by_area=%s, by_name=%s, merged=%s)",
                resolved.by_area,
                resolved.by_name,
                resolved.merged,
            )

            # ---- STEP 2: Ambiguity handling ----
            if not resolved.merged:
                _LOGGER.info("No entities resolved -> Stage 1 clarification")
                return await self._call_stage1_clarification(user_input, result)

            if len(resolved.merged) > 1:
                # Special case: "all lights"/"alle lichter"
                if (
                    entities.get("domain") in ("light", "fan", "cover")
                    and not entities.get("name")
                    and not entities.get("area")
                ):
                    _LOGGER.info(
                        "Multiple matches but global domain request -> executing all"
                    )
                else:
                    _LOGGER.info("Multiple matches -> Stage 1 disambiguation")
                    return await self._call_stage1_disambiguation(
                        user_input, resolved.merged
                    )

            # ---- STEP 3: Execution ----
            if resolved.by_name and not resolved.by_area:
                # Fallback by name → rewrite entities to valid slots
                entities.pop("area", None)
                entities.pop("domain", None)

                ent_reg = entity_registry.async_get(self.hass)
                ent = ent_reg.async_get(resolved.by_name[0])
                friendly_name = (
                    ent.original_name or ent.name or resolved.by_name[0]
                )

                entities.clear()
                entities["name"] = friendly_name
                _LOGGER.debug(
                    "Rewriting entities for execution (fallback by name): %s", entities
                )

                return await self._execute_intent(
                    result.intent.name, entities, language, user_input
                )

            # Normal NLU → let HA handle it
            return await conversation.async_converse(
                self.hass,
                text=utterance,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=language,
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )

        except Exception as err:
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

    # ---- Stage helpers ----
    async def _call_stage1_clarification(self, user_input, resp=None):
        msg = "Stage 1 clarification called"
        _LOGGER.debug(msg)
        return await self._make_placeholder_result(msg, language=user_input.language)

    async def _call_stage1_disambiguation(self, user_input, entity_ids: list[str]):
        msg = f"Stage 1 disambiguation: mehrere Geräte gefunden {entity_ids}"
        _LOGGER.debug(msg)
        return await self._make_placeholder_result(msg, language=user_input.language)

    async def _call_stage2_llm(self, user_input, resp=None):
        msg = "Stage 2 LLM called"
        _LOGGER.debug(msg)
        return await self._make_placeholder_result(msg, language=user_input.language)
