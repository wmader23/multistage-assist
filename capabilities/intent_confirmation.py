import logging
from typing import Any, Dict, List, Optional

from .base import Capability

_LOGGER = logging.getLogger(__name__)

class IntentConfirmationCapability(Capability):
    """Generates a short, natural confirmation sentence."""
    name = "intent_confirmation"
    description = "Generates a natural language confirmation for an action."

    INTENT_DESCRIPTIONS = {
        "HassTurnOn": "The device(s) were turned ON (eingeschaltet/aktiviert/geöffnet).",
        "HassTurnOff": "The device(s) were turned OFF (ausgeschaltet/deaktiviert/geschlossen).",
        "HassLightSet": "Light settings (brightness/color) were adjusted.",
        "HassSetPosition": "Cover/Blind position was set to a specific value.",
        "HassClimateSetTemperature": "Thermostat target temperature was changed.",
        "HassTimerSet": "A timer was successfully set.",
        "HassGetState": "A state or measurement was queried.",
        "HassTemporaryControl": "The device was switched on/off TEMPORARILY for a specific duration.",
    }

    SCHEMA = {
        "properties": {"response": {"type": "string"}},
        "required": ["response"]
    }

    async def run(
        self,
        user_input,
        intent_name: str,
        entity_ids: List[str],
        params: Dict[str, Any] = None,
        **_: Any
    ) -> Dict[str, Any]:
        
        names = []
        domains = []
        states = []
        for eid in entity_ids:
            st = self.hass.states.get(eid)
            if st:
                names.append(st.attributes.get("friendly_name") or eid)
                domains.append(eid.split(".")[0])
                states.append(st.state)
            else:
                names.append(eid)
                domains.append(eid.split(".")[0] if "." in eid else "")
                states.append("unknown")
        
        ignored_keys = {"domain", "service", "entity_id", "area_id"}
        relevant_params = {k: v for k, v in (params or {}).items() if k not in ignored_keys}

        action_desc = self.INTENT_DESCRIPTIONS.get(intent_name, "An action was performed.")
        
        system = f"""You are a smart home assistant.
Generate a VERY SHORT, natural German confirmation (du-form).

## Context
Action: {action_desc}

## Rules
1. **Identify the Device:** Use 'domains' + 'devices' (e.g. domain='light', name='Küche' -> "Das Licht in der Küche").
2. **Be Specific:** Mention duration if available (e.g. "für 10 Minuten").
3. **Be Concise:** No "Okay". Just the facts.

Return JSON: {{"response": "string"}}
"""

        payload = {
            "intent": intent_name,
            "devices": names,
            "domains": domains,
            "states": states,
            "params": relevant_params
        }

        data = await self._safe_prompt({"system": system, "schema": self.SCHEMA}, payload)
        message = data.get("response", "Aktion ausgeführt.") if isinstance(data, dict) else "Aktion ausgeführt."
        
        _LOGGER.debug("[IntentConfirmation] Generated: '%s'", message)
        return {"message": message}