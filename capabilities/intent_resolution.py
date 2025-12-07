import logging
from typing import Any, Dict, List, Optional

from .base import Capability
from .keyword_intent import KeywordIntentCapability
from .entity_resolver import EntityResolverCapability
from .area_alias import AreaAliasCapability
from .memory import MemoryCapability

_LOGGER = logging.getLogger(__name__)

class IntentResolutionCapability(Capability):
    """
    Orchestrates the resolution of a single command string into an Intent + Entities.
    Combines KeywordIntent, EntityResolver, AreaAlias, and Memory logic.
    """

    name = "intent_resolution"
    description = "Resolves a command string to intent and entities."
    
    ENTITY_MATCH_PROMPT = {
        "system": """
You are a smart home helper.
User said a specific device name (e.g. "Spiegellicht") inside a specific area.
Map the user's name to one of the available entities in that area.

## Input
- user_name: The name the user said.
- candidates: List of available entities in the area (id, friendly_name).

## Rules
1. Find the best match based on meaning (e.g. "Spiegellicht" -> "Badezimmer Spiegel").
2. If valid match found, return the entity_id.
3. If no plausible match, return null.

## Output (STRICT)
JSON: {"entity_id": <string or null>}
""",
        "schema": {
            "properties": {"entity_id": {"type": ["string", "null"]}},
            "required": ["entity_id"]
        }
    }

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.keyword_cap = KeywordIntentCapability(hass, config)
        self.resolver_cap = EntityResolverCapability(hass, config)
        self.alias_cap = AreaAliasCapability(hass, config)
        self.memory_cap = MemoryCapability(hass, config)

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        ki_data = await self.keyword_cap.run(user_input)
        intent_name = ki_data.get("intent")
        slots = ki_data.get("slots") or {}

        if not intent_name:
            _LOGGER.debug("[IntentResolution] No intent found.")
            return {}

        entity_ids = []
        name_slot = slots.get("name")
        
        # 2. CHECK MEMORY FOR ENTITY ALIAS (Fast Path)
        if name_slot:
            known_eid = await self.memory_cap.get_entity_alias(name_slot)
            if known_eid:
                if self.hass.states.get(known_eid):
                    _LOGGER.debug("[IntentResolution] Memory hit! Entity '%s' -> %s", name_slot, known_eid)
                    entity_ids = [known_eid]

        # 3. Resolve Entities (Standard)
        if not entity_ids:
            er_data = await self.resolver_cap.run(user_input, entities=slots)
            entity_ids = er_data.get("resolved_ids") or []

        learning_data = None

        # 4. Fallback: Area Alias + Entity Match
        if not entity_ids:
            candidate_area = slots.get("area")
            
            if candidate_area:
                mapped_area = await self.memory_cap.get_area_alias(candidate_area)
                is_new_area = False
                
                if not mapped_area:
                    alias_res = await self.alias_cap.run(user_input, search_text=candidate_area)
                    mapped_area = alias_res.get("area")
                    if mapped_area:
                        is_new_area = True

                if mapped_area:
                    new_slots = slots.copy()
                    if mapped_area == "GLOBAL":
                        new_slots.pop("area", None)
                    else:
                        new_slots["area"] = mapped_area
                    
                    if new_slots.get("name") == candidate_area:
                        new_slots.pop("name")
                    
                    er_data = await self.resolver_cap.run(user_input, entities=new_slots)
                    entity_ids = er_data.get("resolved_ids") or []
                    
                    # --- ENTITY ALIAS LEARNING (The Fix) ---
                    if not entity_ids and name_slot and mapped_area != "GLOBAL":
                        _LOGGER.debug("[IntentResolution] Trying to match name '%s' within area '%s'", name_slot, mapped_area)
                        domain = new_slots.get("domain")
                        area_candidates = self.resolver_cap._entities_in_area_by_name(mapped_area, domain)
                        
                        if area_candidates:
                            payload = {"user_name": name_slot, "candidates": area_candidates}
                            match_res = await self._safe_prompt(self.ENTITY_MATCH_PROMPT, payload)
                            matched_eid = match_res.get("entity_id")
                            
                            if matched_eid and self.hass.states.get(matched_eid):
                                entity_ids = [matched_eid]
                                learning_data = {
                                    "type": "entity",
                                    "source": name_slot,
                                    "target": matched_eid
                                }
                                _LOGGER.debug("[IntentResolution] LLM matched '%s' -> %s", name_slot, matched_eid)

                    if entity_ids and is_new_area and not learning_data and mapped_area != "GLOBAL":
                         learning_data = {
                             "type": "area",
                             "source": candidate_area,
                             "target": mapped_area
                         }
                    slots = new_slots

        if not entity_ids:
            _LOGGER.debug("[IntentResolution] Failed to resolve entities for intent %s", intent_name)
            return {}

        return {
            "intent": intent_name,
            "slots": slots,
            "entity_ids": entity_ids,
            "learning_data": learning_data
        }