import logging
from typing import Any, Dict, List

from homeassistant.components import conversation
from homeassistant.helpers import intent

from .prompt_executor import PromptExecutor
from .prompts import (
    CLARIFICATION_PROMPT,
    CLARIFICATION_PROMPT_STAGE2,
    ENTITY_FILTER_PROMPT,
    PLURAL_SINGULAR_PROMPT,
    DISAMBIGUATION_PROMPT,
)
from .stage0 import Stage0Result
from .conversation_utils import make_response

_LOGGER = logging.getLogger(__name__)


def _with_new_text(user_input: conversation.ConversationInput, new_text: str) -> conversation.ConversationInput:
    """Clone ConversationInput with modified text."""
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        satellite_id=getattr(user_input, "satellite_id", None),
        language=user_input.language,
        agent_id=getattr(user_input, "agent_id", None),
        extra_system_prompt=getattr(user_input, "extra_system_prompt", None),
    )


class Stage1Processor:
    """Stage 1: Clarification, disambiguation, plural detection, and filtering."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _apply_filter_hints(self, candidates: List[str], hints: Dict[str, Any]) -> List[str]:
        if not isinstance(hints, dict):
            return candidates

        name_f = (hints.get("name") or "").lower()
        dc_f = (hints.get("device_class") or "").lower()
        unit_f = (hints.get("unit") or "").lower()
        area_f = (hints.get("area") or "").lower()
        domain_f = (hints.get("domain") or "").lower()
        must_include: List[str] = [s.lower() for s in hints.get("must_include", []) if isinstance(s, str)]
        must_exclude: List[str] = [s.lower() for s in hints.get("must_exclude", []) if isinstance(s, str)]

        filtered: List[str] = []
        for eid in candidates:
            state = self.hass.states.get(eid)
            if not state:
                continue
            attrs = state.attributes or {}
            fname = (attrs.get("friendly_name") or "").lower()
            dev_class = (attrs.get("device_class") or "").lower()
            unit = (attrs.get("unit_of_measurement") or "").lower()
            area = (attrs.get("area_id") or attrs.get("area") or "").lower()
            domain = eid.split(".", 1)[0].lower()

            if name_f and name_f not in fname and name_f not in eid.lower():
                continue
            if dc_f and dc_f != dev_class:
                continue
            if unit_f and unit_f not in unit:
                continue
            if area_f and area_f not in area and area_f not in fname and area_f not in eid.lower():
                continue
            if domain_f and domain_f != domain:
                continue
            if must_include and not all(any(token in s for s in (eid.lower(), fname)) for token in must_include):
                continue
            if any(token in eid.lower() or token in fname for token in must_exclude):
                continue
            filtered.append(eid)
        return filtered

    async def _is_plural(self, text: str) -> bool:
        data = await self.prompts.run(PLURAL_SINGULAR_PROMPT, {"user_input": text})
        is_plural = bool(data and data.get("multiple_entities"))
        _LOGGER.debug("Plural detection for '%s' -> %s (raw=%s)", text, is_plural, data)
        return is_plural

    async def _call_action_disambiguation(self, user_input, entity_ids: List[str], intent_obj=None):
        """Ask the user to clarify which device was meant."""
        entity_map = {eid: (self.hass.states.get(eid).attributes.get("friendly_name") if self.hass.states.get(eid) else eid) for eid in entity_ids}
        _LOGGER.debug("Stage1 disambiguation candidates: %s", entity_map)

        candidates_ordered = [{"entity_id": eid, "name": name} for eid, name in entity_map.items()]
        data = await self.prompts.run(DISAMBIGUATION_PROMPT, {"input_entities": candidates_ordered})
        msg = (data or {}).get("message") or "Bitte präzisiere, welches Gerät du meinst."

        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        self._pending[key] = {"kind": "action", "intent": intent_obj, "candidates": entity_map}
        return await make_response(msg, user_input)

    async def run(self, user_input: conversation.ConversationInput, raw_stage0: Stage0Result | None = None):
        """Main entrypoint for Stage 1 logic."""
        _LOGGER.debug("Stage1 clarification for input: %s", user_input.text)

        # --- Handle multiple resolved entities here ---
        if raw_stage0 and raw_stage0.type == "intent" and len(raw_stage0.resolved_ids or []) > 1:
            _LOGGER.debug("Multiple entities found → initiating disambiguation in Stage1")
            return await self._call_action_disambiguation(
                user_input, raw_stage0.resolved_ids, intent_obj=raw_stage0.intent
            )

        # --- Regular clarification fallback ---
        data = await self.prompts.run(CLARIFICATION_PROMPT, {"user_input": user_input.text})
        _LOGGER.debug("Stage1 clarification result: %s", data)

        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            if len(data) == 1 and data[0].strip() == user_input.text.strip():
                _LOGGER.info("Stage1-A returned identical text → delegate to Stage1-B")
            else:
                results = []
                for clarified_command in data:
                    clarified_input = _with_new_text(user_input, clarified_command)
                    res = await conversation.async_converse(
                        self.hass,
                        text=clarified_input.text,
                        context=clarified_input.context,
                        conversation_id=clarified_input.conversation_id,
                        language=clarified_input.language or "de",
                        agent_id=conversation.HOME_ASSISTANT_AGENT,
                    )
                    results.append(res)
                return results[-1] if results else await make_response(
                    "Entschuldigung, ich konnte das nicht verarbeiten.", user_input
                )

        _LOGGER.debug("Stage1-A insufficient → Stage1-B")
        data2 = await self.prompts.run(CLARIFICATION_PROMPT_STAGE2, {"user_input": user_input.text})
        _LOGGER.debug("Stage1-B clarification result: %s", data2)

        if not isinstance(data2, dict):
            _LOGGER.error("Stage1-B returned invalid format: %s", data2)
            return await make_response("Entschuldigung, ich konnte deine Anweisung nicht verstehen.", user_input)

        # Default: pass through to HA agent
        return await conversation.async_converse(
            self.hass,
            text=user_input.text,
            context=user_input.context,
            conversation_id=user_input.conversation_id,
            language=user_input.language or "de",
            agent_id=conversation.HOME_ASSISTANT_AGENT,
        )
