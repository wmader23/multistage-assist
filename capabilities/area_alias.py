import logging
from typing import Any, Dict, List

from homeassistant.helpers import area_registry as ar
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class AreaAliasCapability(Capability):
    """
    LLM-based mapping from a *German* user utterance to one of the existing
    Home Assistant areas.

    The model sees:
    - the full German user command (e.g. "Schalte das Licht im Bad an")
    - the list of HA areas (e.g. ["Badezimmer", "Wohnzimmer"])

    It must decide whether the utterance refers to one of them.
    """

    name = "area_alias"
    description = "Map a German user utterance to a Home Assistant area."

    PROMPT = {
        "system": """
You are an assistant that maps a *German* smart-home voice command
to one of the Home Assistant areas.

## Task
Given:
- the full German user utterance (e.g. "Schalte das Licht im Bad an")
- the list of existing HA areas

Your job:
1. Determine whether the utterance refers to one of these areas.
2. Output ONLY an area name from the list — never invent or modify names.
3. If none of the areas match, return null.

You MUST analyze the entire utterance (not just single words).
You MUST handle abbreviations, synonyms, and natural language
(e.g. "Bad" → "Badezimmer").

## German examples (IMPORTANT: keep these in German)
- utterance: "Schalte das Licht im Bad an"
  areas: ["Wohnzimmer","Badezimmer"]
  → area = "Badezimmer"

- utterance: "Mach das Licht im Kinderbad an"
  areas: ["Kinderzimmer","Kinder Badezimmer"]
  → area = "Kinder Badezimmer"

- utterance: "Schalte das Licht im Garten an"
  areas: ["Wohnzimmer","Badezimmer"]
  → area = null

## Output format (STRICT)
Return a JSON object:
{ "area": <string or null> }

No explanations. No additional fields.
""",
        "schema": {
            "type": "object",
            "properties": {
                "area": {
                    "type": ["string", "null"],
                }
            },
            "required": ["area"],
            "additionalProperties": False,
        },
    }

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        """Run the area alias LLM mapping."""
        text = (user_input.text or "").strip()
        if not text:
            return {"area": None}

        # Load HA areas
        area_reg = ar.async_get(self.hass)
        areas: List[str] = [a.name for a in area_reg.async_list_areas() if a.name]

        if not areas:
            _LOGGER.debug("[AreaAlias] No areas defined → cannot map.")
            return {"area": None}

        payload = {
            "utterance": text,   # full German utterance
            "areas": areas,      # all available HA area names
        }

        _LOGGER.debug(
            "[AreaAlias] Mapping utterance=%r to one of areas=%r",
            text,
            areas,
        )

        data = await self._safe_prompt(self.PROMPT, payload)

        if not isinstance(data, dict):
            _LOGGER.warning("[AreaAlias] Model returned invalid object: %r", data)
            return {"area": None}

        area = data.get("area")

        if not isinstance(area, str) or not area.strip():
            _LOGGER.debug("[AreaAlias] No valid area chosen.")
            return {"area": None}

        area = area.strip()

        if area not in areas:
            _LOGGER.debug(
                "[AreaAlias] Model chose %r which is not in available areas %r → ignoring.",
                area,
                areas,
            )
            return {"area": None}

        _LOGGER.debug("[AreaAlias] Successfully mapped → %r", area)
        return {"area": area}
