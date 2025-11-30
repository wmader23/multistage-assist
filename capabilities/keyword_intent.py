import logging
from typing import Any, Dict, Optional, List

from .base import Capability
from .plural_detection import LIGHT_KEYWORDS, COVER_KEYWORDS, SENSOR_KEYWORDS, CLIMATE_KEYWORDS

_LOGGER = logging.getLogger(__name__)


class KeywordIntentCapability(Capability):
    """
    Detect a domain from German keywords and let the LLM pick
    a specific Home Assistant intent + slots within that domain.
    """

    name = "keyword_intent"
    description = "Derive a Home Assistant intent from a single German command using keyword domains."

    # Domain → list of keywords (singular + plural) reusing plural_detection helpers.
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "light": list(LIGHT_KEYWORDS.keys()) + list(LIGHT_KEYWORDS.values()),
        "cover": list(COVER_KEYWORDS.keys()) + list(COVER_KEYWORDS.values()),
        "sensor": list(SENSOR_KEYWORDS.keys()) + list(SENSOR_KEYWORDS.values()) + ["grad", "warm", "kalt", "wieviel"],
        "climate": list(CLIMATE_KEYWORDS.keys()) + list(CLIMATE_KEYWORDS.values()) + ["klima"],
    }

    # Intents per domain + extra description + examples to tune the prompt.
    INTENT_DOMAINS: Dict[str, Dict[str, Any]] = {
        "light": {
            "description": "Steuerung von Lichtern, Lampen und Beleuchtung.",
            "intents": {
                "HassTurnOn": "Schaltet eine oder mehrere Lichter/Lampen ein.",
                "HassTurnOff": "Schaltet eine oder mehrere Lichter/Lampen aus.",
                "HassLightSet": "Setzt Helligkeit (brightness) oder Farbe eines Lichts.",
                "HassGetState": "Fragt den Zustand eines Lichts ab.",
            },
            "slot_rules": """
- 'brightness': Integer 0-100 OR relative command string.
  - If user gives a specific number -> use the integer (e.g., 50).
  - If user says "dimmen", "dunkler", "weniger hell" without a number -> use string "step_down"
  - If user says "heller", "aufhellen", "mehr licht" without a number -> use string "step_up"
- 'color_name': If user specifies a color (e.g. 'rot', 'blau').
- DO NOT use generic slots like 'value' or 'action'.
            """,
            "examples": [
                'User: "Schalte das Licht in der Dusche an"\n'
                '→ {"intent":"HassTurnOn","slots":{"name":"Dusche","domain":"light"}}',
                'User: "Dimme das Licht im Wohnzimmer"\n'
                '→ {"intent":"HassLightSet","slots":{"area":"Wohnzimmer","domain":"light","brightness":"step_down"}}',
                'User: "Mach das Licht heller"\n'
                '→ {"intent":"HassLightSet","slots":{"domain":"light","brightness":"step_up"}}',
            ],
        },
        "cover": {
            "description": "Steuerung von Rollläden, Rollos und Jalousien.",
            "intents": {
                "HassTurnOn": "Öffnet Rollläden (hochfahren).",
                "HassTurnOff": "Schließt Rollläden (runterfahren).",
                "HassSetPosition": "Setzt die Position eines Rollladens (0–100%).",
                "HassGetState": "Fragt den Zustand eines Rollladens ab.",
            },
            "examples": [
                'User: "Fahre den Rollladen im Wohnzimmer ganz runter"\n'
                '→ {"intent":"HassSetPosition","slots":{"area":"Wohnzimmer","position":0,"domain":"cover"}}',
            ],
        },
        "sensor": {
            "description": "Abfragen von Sensorwerten (Temperatur, Feuchtigkeit, Status).",
            "intents": {
                "HassGetState": "Fragt den Wert oder Zustand eines Sensors ab.",
            },
            "examples": [
                'User: "Wie ist die Temperatur im Büro?"\n'
                '→ {"intent":"HassGetState","slots":{"area":"Büro","domain":"sensor", "name":"Temperatur"}}',
            ],
        },
        "climate": {
            "description": "Steuerung von Heizkörpern, Thermostaten und Klimaanlagen.",
            "intents": {
                "HassClimateSetTemperature": "Setzt die Zieltemperatur.",
                "HassTurnOn": "Schaltet Heizung/Klima ein.",
                "HassTurnOff": "Schaltet Heizung/Klima aus.",
                "HassGetState": "Fragt Status der Heizung ab.",
            },
            "examples": [
                'User: "Stelle die Heizung im Bad auf 22 Grad"\n'
                '→ {"intent":"HassClimateSetTemperature","slots":{"area":"Bad","temperature":22}}',
            ],
        },
    }

    # JSON-Schema for PromptExecutor/_safe_prompt
    SCHEMA: Dict[str, Any] = {
        "properties": {
            "intent": {"type": ["string", "null"]},
            "slots": {"type": "object"},
        },
    }

    # ... (rest of the file remains unchanged: _detect_domain, _build_system_prompt, run) ...
    def _detect_domain(self, text: str) -> Optional[str]:
        """Keyword-based (with fuzzy) domain detection."""
        t = (text or "").lower()
        matches: List[str] = []

        # 1) Exact substring matching
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(k in t for k in keywords):
                matches.append(domain)

        if len(matches) == 1:
            _LOGGER.debug("[KeywordIntent] Detected domain '%s' (exact)", matches[0])
            return matches[0]

        if matches:
            if "climate" in matches and "sensor" in matches:
                 return "climate"
            _LOGGER.debug("[KeywordIntent] Ambiguous matches: %s", matches)
            return None

        # 2) Fuzzy matching fallback
        import difflib
        import re
        tokens = re.findall(r"\w+", t)
        fuzzy_scores: Dict[str, float] = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            best_score = 0.0
            for kw in keywords:
                kw_l = kw.lower()
                for tok in tokens:
                    score = difflib.SequenceMatcher(None, kw_l, tok).ratio()
                    if score > best_score:
                        best_score = score
            if best_score >= 0.8:
                fuzzy_scores[domain] = best_score

        if len(fuzzy_scores) == 1:
            domain = next(iter(fuzzy_scores.keys()))
            _LOGGER.debug("[KeywordIntent] Detected domain '%s' (fuzzy)", domain)
            return domain

        return None

    def _build_system_prompt(self, domain: str, meta: Dict[str, Any]) -> str:
        desc = meta.get("description") or ""
        intents: Dict[str, str] = meta.get("intents") or {}
        examples: List[str] = meta.get("examples") or []
        slot_rules = meta.get("slot_rules") or ""

        lines: List[str] = [
            "You are a language model that selects a Home Assistant intent.",
            f"Current domain: {domain}",
            f"Description: {desc}",
            "",
            "Allowed intents:",
        ]
        for iname, idesc in intents.items():
            lines.append(f"- {iname}: {idesc}")

        lines += [
            "",
            "Slots hints:",
            "- 'area': Room name (e.g. 'Büro').",
            "- 'name': Device name.",
            "- 'domain': 'light', 'cover', 'sensor', 'climate'.",
        ]
        
        if slot_rules:
            lines.append(slot_rules)

        lines.append("")
        lines.append("Examples:")
        for ex in examples:
            lines.append(ex)

        return "\n".join(lines)

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text or ""
        domain = self._detect_domain(text)

        if not domain:
            return {}

        meta = self.INTENT_DOMAINS.get(domain)
        if not meta:
            return {}

        system = self._build_system_prompt(domain, meta)
        prompt = {
            "system": system,
            "schema": self.SCHEMA,
        }

        data = await self._safe_prompt(prompt, {"user_input": text})

        if not isinstance(data, dict):
            return {}

        intent = data.get("intent")
        if not intent:
            return {}

        slots = data.get("slots") or {}
        if "domain" not in slots:
             slots["domain"] = domain

        return {
            "domain": domain,
            "intent": intent,
            "slots": slots,
        }
