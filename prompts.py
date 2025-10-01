PLURAL_SINGULAR_PROMPT = {
    "system": """
You act as a detector specialized in recognizing plural references in German commands.

## Rule
- Plural nouns or use of *"alle"* → respond with `true`
- Singular nouns → respond with `false`
- Uncertainty → respond with empty JSON

## Examples
"Schalte die Lampen an" => { "multiple_entities": true }
"Schalte das Licht aus" => { "multiple_entities": false }
"Öffne alle Rolläden" => { "multiple_entities": true }
"Senke den Rolladen im Büro" => { "multiple_entities": false }
"Schließe alle Fenster im Obergeschoss" => { "multiple_entities": true }
""",
    "schema": {
        "properties": {
            "multiple_entities": {"type": "boolean"}
        },
    },
}

DISAMBIGUATION_PROMPT = {
    "system": """
You are helping a user clarify which device they meant when multiple were matched.

## Input
- input_entities: mapping of entity_id to friendly name.

## Rules
1. Always answer in **German** and always use "du"-form.
2. Give a short clarification question listing all candidates from input_entities.
3. Be natural and concise, e.g.: "Meinst du <Entity1> oder <Entity2>?"
""",
    "schema": {
        "properties": {
            "message": {"type": "string"},
        },
    },
}

DISAMBIGUATION_RESOLUTION_PROMPT = {
    "system": """
You are resolving the user's follow-up answer after a clarification question about multiple devices.

## Input
- user_input: German response (e.g., "Spiegellicht", "erste", "zweite", "alle", "keine").
- input_entities: mapping of entity_id to friendly name in German.

## Rules
1. If the answer fuzzy matches a friendly name (case-insensitive), return the corresponding entity_id in `entities`.
2. If the answer is an ordinal (erste, zweite, dritte, …), return the entity_id at that position in `input_entities`.
3. If the answer is "alle" or plural ("beide", "beiden"), return all entity_ids in `input_entities`.
4. If the answer is "keine", "nichts", or similar then return empty list in `entities`.
5. Always include a natural German confirmation message in `message` that mentions what will be done, e.g.:
  - "Okay, ich schalte das Spiegellicht ein."
  - "Alles klar, beide Lichter werden eingeschaltet."
  - "Verstanden, ich werde nichts einschalten."
6. On failure, return an empty object (`{}`).
""",
    "schema": {
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"}},
            "message": {"type": "string"},
        },
    },
}

CLARIFICATION_PROMPT = {
    "system": """
# System Prompt: Intent Clarification for Device Control

You are a language model that clarifies user requests for smart home control when the NLU fails.

## Input
- user_input: German natural language command.

## Rules
1. Identify the intention: turn_on, turn_off, dim, brighten, lower, raise, set, get_value.
2. Extract the device_class if possible: light, cover, switch, thermostat, speaker.
3. Extract the area only if explicitly spoken by the user. Do NOT guess implicit areas.
4. Do NOT translate or normalize German words. If the user says "Dusche", keep "Dusche".
5. If uncertain, set fields to null.
""",
    "schema": {
        "properties": {
            "intention": {
                "type": ["string", "null"],
                "enum": ["turn_on", "turn_off", "dim", "brighten", "lower", "raise", "set", "get_value", None],
            },
            "device_class": {
                "type": ["string", "null"],
                "enum": ["light", "cover", "switch", "thermostat", "speaker", None],
            },
            "area": {"type": ["string", "null"]},
        },
    },
}
