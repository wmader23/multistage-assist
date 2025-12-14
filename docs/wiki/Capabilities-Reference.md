# Capabilities Reference

Complete reference of all available capabilities in MultiStage Assist.

## Core Capabilities

### clarification

**Purpose:** Split compound commands and transform implicit commands.

**Input:** User input text

**Output:** Array of atomic commands

**Features:**
- Splits "und" compounds
- Converts "zu dunkel" → "Licht heller"
- Preserves durations
- Handles multi-area commands

**Example:**
```
Input: "Licht an und Rollo runter"
Output: ["Schalte Licht an", "Fahre Rollo runter"]
```

---

### keyword_intent

**Purpose:** Detect domain and extract intent from keywords.

**Input:** User input text

**Output:** `{domain, intent, slots}`

**Supported Domains:**
- light, cover, switch, fan
- climate, sensor, media_player
- timer, vacuum, calendar, automation

**Example:**
```
Input: "Licht im Bad auf 50%"
Output: {
  domain: "light",
  intent: "HassLightSet",
  slots: {area: "Bad", brightness: "50"}
}
```

---

### intent_resolution

**Purpose:** Resolve entities from slots using fuzzy matching.

**Input:** User input, keyword_intent data

**Output:** `{intent, slots, entity_ids, learning_data}`

**Features:**
- Area alias resolution
- Floor detection
- Fuzzy entity name matching

---

### entity_resolver

**Purpose:** Find entity IDs matching criteria.

**Input:** Domain, area, floor, name filters

**Output:** List of entity IDs

**Features:**
- Exposure filtering
- Domain filtering
- Area/floor/name matching

---

### intent_executor

**Purpose:** Execute Home Assistant intents.

**Input:** User input, intent name, entity IDs, params

**Output:** Conversation result

**Features:**
- Brightness step up/down (relative %)
- Timebox calls for temporary controls
- State verification after execution
- Error handling

**Brightness Logic:**
- step_up: +20% of current (min 5%)
- step_down: -20% of current (min 5%)
- From 0%: turns on to 30%

---

### intent_confirmation

**Purpose:** Generate natural language confirmation.

**Input:** Intent, devices, params

**Output:** German confirmation text

**Example:**
```
Intent: HassLightSet
Devices: ["Büro"]
Params: {brightness: 50}
Output: "Das Licht im Büro ist auf 50% gesetzt."
```

---

### disambiguation

**Purpose:** Help user select from ambiguous entities.

**Input:** User input, candidate entities

**Output:** Narrowed candidates or selection

---

### disambiguation_select

**Purpose:** Handle user's disambiguation choice.

**Input:** User response, candidates

**Output:** Selected entity

---

## Domain Capabilities

### calendar

**Purpose:** Create calendar events.

**Input:** User input, intent, slots

**Output:** Event creation result or follow-up question

**Features:**
- Multi-turn conversation
- Relative date resolution
- Time range parsing
- Calendar selection
- Confirmation flow

**Date Patterns:**
- heute, morgen, übermorgen
- in X Tagen
- nächsten Montag
- am 25. Dezember

---

### timer

**Purpose:** Set timers with notifications.

**Input:** User input, intent, slots

**Output:** Timer creation result

**Features:**
- Duration parsing
- Named timers
- Device notifications
- Memory integration for devices

---

## Utility Capabilities

### memory

**Purpose:** Store and retrieve learned aliases.

**Input:** Get/set operations

**Output:** Stored data

**Storage:**
- Area aliases: "bad" → "Badezimmer"
- Entity aliases: "daniels handy" → "notify.mobile_app_..."
- Floor mappings

---

### area_alias

**Purpose:** Fuzzy match area names.

**Input:** User query, candidate areas

**Output:** Best matching area

**Features:**
- Synonym handling (Bad → Badezimmer)
- Global scope detection (Haus, Wohnung)
- LLM-powered matching

---

### plural_detection

**Purpose:** Detect if user is referring to multiple entities.

**Input:** User text, domain

**Output:** Boolean is_plural

**Examples:**
- "alle Lichter" → plural
- "das Licht" → singular

---

### yes_no_response

**Purpose:** Generate yes/no answers for state queries.

**Input:** User input, domain, state, entity_ids

**Output:** Yes/no response text

---

### chat

**Purpose:** Free-form conversation fallback.

**Input:** User input, context

**Output:** Conversational response

---

## Capability Configuration

### Prompt Schema

Each LLM-based capability defines:

```python
PROMPT = {
    "system": "System instruction for LLM",
    "schema": {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"}
        }
    }
}
```

### Registration

Capabilities are registered in stages:

```python
# stage1.py
self.register(ClarificationCapability(hass, llm_config))
self.register(KeywordIntentCapability(hass, llm_config))
# ... etc
```

### Usage

```python
# Use capability and get result
result = await self.use("capability_name", user_input)

# Get capability instance
cap = self.get("capability_name")
await cap.run(user_input, **params)
```
