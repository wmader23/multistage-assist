import dataclasses
from homeassistant.helpers import area_registry, entity_registry
from rapidfuzz import fuzz


@dataclasses.dataclass
class ResolvedEntities:
    by_area: list[str]
    by_name: list[str]

    @property
    def merged(self) -> list[str]:
        return list({*self.by_area, *self.by_name})


class EntityResolver:
    """Utility to resolve entities from NLU slots with fuzzy matching."""

    def __init__(self, hass, threshold: int = 90):
        self.hass = hass
        self.threshold = threshold  # min similarity percentage

    async def resolve(self, entities: dict[str, str]) -> ResolvedEntities:
        ent_reg = entity_registry.async_get(self.hass)
        area_reg = area_registry.async_get(self.hass)

        domain = entities.get("domain")
        area_name = entities.get("area")
        name = entities.get("name")

        by_area: list[str] = []
        by_name: list[str] = []

        norm_area = area_name.strip().lower() if area_name else None
        norm_name = name.strip().lower() if name else None

        for ent in ent_reg.entities.values():
            if domain and ent.domain != domain:
                continue

            # area matching (fuzzy)
            if norm_area and ent.area_id:
                area = area_reg.async_get_area(ent.area_id)
                if area:
                    score = fuzz.ratio(norm_area, area.name.lower())
                    if score >= self.threshold:
                        by_area.append(ent.entity_id)

            # name matching (fuzzy)
            if norm_name and ent.original_name:
                score = fuzz.ratio(norm_name, ent.original_name.lower())
                if score >= self.threshold:
                    by_name.append(ent.entity_id)

            # fallback: only domain
            if domain and not norm_area and not norm_name:
                by_area.append(ent.entity_id)

            # fallback: only area
            if norm_area and not domain and not norm_name and ent.area_id:
                area = area_reg.async_get_area(ent.area_id)
                if area:
                    score = fuzz.ratio(norm_area, area.name.lower())
                    if score >= self.threshold:
                        by_area.append(ent.entity_id)

        return ResolvedEntities(by_area=by_area, by_name=by_name)

    async def make_entity_map(self, entity_ids: list[str]) -> dict[str, str]:
        ent_reg = entity_registry.async_get(self.hass)
        states = self.hass.states

        entity_map = {}
        for eid in entity_ids:
            ent = ent_reg.async_get(eid)
            state = states.get(eid)
            if state and state.name:
                entity_map[eid] = state.name
            elif ent and ent.original_name:
                entity_map[eid] = ent.original_name
            else:
                entity_map[eid] = eid

        return entity_map
