import logging
from typing import Any, Dict, List, Optional

from .base import Capability

_LOGGER = logging.getLogger(__name__)


class IntentConfirmationCapability(Capability):
    """
    Generates a short, natural confirmation sentence using an LLM.
    Dynamically builds the prompt based on the executed intent.
    """

    name = "intent_confirmation"
    description = "Generates a natural language confirmation for an action."

    # Map intents to a clear, natural description of what just happened.
    # This guides the LLM to generate the correct confirmation without needing many examples.
    INTENT_DESCRIPTIONS = {
        "HassTurnOn": "The device(s) were turned ON (eingeschaltet/aktiviert/geöffnet).",
        "HassTurnOff": "The device(s) were turned OFF (ausgeschaltet/deaktiviert/geschlossen).",
        "HassLightSet": "Light settings (brightness/color) were adjusted.",
        "HassSetPosition": "Cover/Blind position was set to a specific value.",
        "HassClimateSetTemperature": "Thermostat target temperature was changed.",
        "HassTimerSet": "A timer was successfully set.",
        "HassGetState": "A state or measurement was queried.",
    }

    SCHEMA = {
        "properties": {
            "response": {"type": "string"}
        },
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
        
        # 1. Resolve Friendly Names & Domains
        names = []
        domains = []
        states = [] # Optional, but helpful context if available

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
        
        # 2. Filter Parameters
        ignored_keys = {"domain", "service", "entity_id", "area_id"}
        relevant_params = {k: v for k, v in (params or {}).items() if k not in ignored_keys}

        # 3. Build Dynamic System Prompt
        action_desc = self.INTENT_DESCRIPTIONS.get(intent_name, "An action was performed on the device.")
        
        system = f"""You are a smart home assistant.
Generate a VERY SHORT, natural German confirmation (du-form).

## Context
Action: {action_desc}

## Rules
1. **Identify the Device:** Use the 'domains' list to identify what the device is (e.g. domain='light' -> "Licht", 'cover' -> "Rollladen").
2. **Be Specific:** Combine the device type with the name (e.g. "Küche" + light -> "Das Licht in der Küche").
3. **Be Concise:** Do not say "Okay" or "Erledigt". Just describe the new state.

Return JSON: {{"response": "string"}}
"""

        # 4. Prepare Payload
        payload = {
            "intent": intent_name,
            "devices": names,
            "domains": domains,
            "params": relevant_params,
            # We pass states if useful, though intent description implies the target state
            # "current_states": states 
        }

        # 5. Generate
        data = await self._safe_prompt({"system": system, "schema": self.SCHEMA}, payload)
        
        message = "Aktion ausgeführt."
        if isinstance(data, dict):
            message = data.get("response", message)

        _LOGGER.debug("[IntentConfirmation] Generated: '%s'", message)
        return {"message": message}