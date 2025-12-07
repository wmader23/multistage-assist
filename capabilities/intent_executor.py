import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import intent as ha_intent
from homeassistant.core import Context
from homeassistant.components.conversation import ConversationResult

from .base import Capability

_LOGGER = logging.getLogger(__name__)


class IntentExecutorCapability(Capability):
    """
    Execute a known HA intent for one or more concrete entity_ids.
    """

    name = "intent_executor"
    description = "Execute a Home Assistant intent for specific targets."
    
    RESOLUTION_KEYS = {"area", "room", "floor", "device_class"}
    BRIGHTNESS_STEP = 20

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
            return {}

        hass = self.hass
        params = params or {}

        valid_ids = [eid for eid in entity_ids if hass.states.get(eid) and hass.states.get(eid).state not in ("unavailable", "unknown")]
        if not valid_ids:
            return {}

        results: List[tuple[str, ha_intent.IntentResponse]] = []
        final_executed_params = params.copy()
        
        for eid in valid_ids:
            # Determine effective intent
            effective_intent = intent_name
            domain = eid.split(".", 1)[0]
            current_params = params.copy()

            if intent_name == "HassClimateGetTemperature" and domain == "sensor":
                effective_intent = "HassGetState"

            # Relative Brightness Logic
            if intent_name == "HassLightSet" and "brightness" in current_params:
                val = current_params["brightness"]
                if val in ("step_up", "step_down"):
                    state_obj = hass.states.get(eid)
                    if state_obj:
                        cur_255 = state_obj.attributes.get("brightness") or 0
                        cur_pct = int((cur_255 / 255.0) * 100)
                        
                        if val == "step_up":
                            new_pct = min(100, cur_pct + self.BRIGHTNESS_STEP)
                            if cur_pct == 0: new_pct = self.BRIGHTNESS_STEP
                        else:
                            new_pct = max(0, cur_pct - self.BRIGHTNESS_STEP)
                        
                        current_params["brightness"] = new_pct
                        final_executed_params["brightness"] = new_pct
                    else:
                        current_params.pop("brightness")

            # Slots
            slots = {"name": {"value": eid}}
            if "domain" not in current_params:
                slots["domain"] = {"value": domain}

            for k, v in current_params.items():
                if k in self.RESOLUTION_KEYS or k == "name":
                    continue
                slots[k] = {"value": v}

            _LOGGER.debug("[IntentExecutor] Executing %s on %s", effective_intent, eid)

            try:
                resp = await ha_intent.async_handle(
                    hass,
                    platform="conversation",
                    intent_type=str(effective_intent),
                    slots=slots,
                    text_input=user_input.text,
                    context=user_input.context or Context(),
                    language=language or (user_input.language or "de"),
                )
                results.append((eid, resp))
            except Exception as e:
                _LOGGER.warning("[IntentExecutor] Error on %s: %s", eid, e)

        if not results:
            return {}

        final_resp = results[-1][1]

        # Ensure minimal speech so UI doesn't hang, but let CommandProcessor override it
        def _has_speech(r):
            s = getattr(r, "speech", None)
            return isinstance(s, dict) and bool(s.get("plain", {}).get("speech"))

        if not _has_speech(final_resp):
            final_resp.async_set_speech("Okay.")

        return {
            "result": ConversationResult(
                response=final_resp,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            ),
            "executed_params": final_executed_params
        }