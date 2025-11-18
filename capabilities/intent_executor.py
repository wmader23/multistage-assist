import logging
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation
from homeassistant.helpers import intent as ha_intent

from .base import Capability

_LOGGER = logging.getLogger(__name__)


def _join_names(names: List[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} und {names[1]}"
    return f"{', '.join(names[:-1])} und {names[-1]}"


class IntentExecutorCapability(Capability):
    """
    Execute a known HA intent for one or more concrete entity_ids by calling
    homeassistant.helpers.intent.async_handle directly.
    Ensures a plain speech fallback so the UI doesn't crash.
    """

    name = "intent_executor"
    description = "Execute a Home Assistant intent for specific targets and return a ConversationResult."

    async def run(
        self,
        user_input,
        *,
        intent_name: str,
        entity_ids: List[str],
        params: Optional[Dict[str, Any]] = None,
        language: str = "de",
        **_: Any,
    ) -> Dict[str, Any]:
        if not intent_name or not entity_ids:
            _LOGGER.warning(
                "[IntentExecutor] Missing intent_name or entity_ids (intent=%r, entities=%r)",
                intent_name, entity_ids,
            )
            return {}

        # Execute once per entity
        responses: List[ha_intent.IntentResponse] = []
        for eid in entity_ids:
            slots = {"name": {"value": eid}}
            if params:
                for k, v in params.items():
                    if k == "name":
                        continue
                    slots[k] = {"value": v}

            _LOGGER.debug(
                "[IntentExecutor] Executing intent '%s' with slots=%s (text=%r)",
                intent_name, slots, user_input.text,
            )
            resp = await ha_intent.async_handle(
                self.hass,
                platform="conversation",
                intent_type=str(intent_name),
                slots=slots,
                text_input=user_input.text,
                context=user_input.context,
                language=language or (user_input.language or "de"),
            )
            responses.append(resp)

        if not responses:
            _LOGGER.warning("[IntentExecutor] No responses returned from async_handle()")
            return {}

        def _has_plain_speech(r: ha_intent.IntentResponse) -> bool:
            s = getattr(r, "speech", None)
            if not isinstance(s, dict):
                return False
            plain = s.get("plain") or {}
            return bool(plain.get("speech"))

        # Prefer last response that already has speech
        final_resp = next((r for r in reversed(responses) if _has_plain_speech(r)), responses[-1])

        # Fallback speech so the frontend never explodes
        if not _has_plain_speech(final_resp):
            friendly: List[str] = []
            for eid in entity_ids:
                st = self.hass.states.get(eid)
                name = (st and st.attributes.get("friendly_name")) or eid
                friendly.append(name)
            names = _join_names(friendly)

            # Keep this lean & generic; no LLM here
            msg = "Okay."

            final_resp.async_set_speech(msg)

        conv_result = conversation.ConversationResult(
            response=final_resp,
            conversation_id=user_input.conversation_id,
            continue_conversation=False,
        )
        return {"result": conv_result}
