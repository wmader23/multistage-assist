"""Calendar capability using MultiTurnCapability base class."""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant
from homeassistant.components import conversation

from .multi_turn_base import MultiTurnCapability
from custom_components.multistage_assist.conversation_utils import make_response
from ..utils.fuzzy_utils import fuzzy_match_candidates


_LOGGER = logging.getLogger(__name__)


class CalendarCapability(MultiTurnCapability):
    """Create calendar events on Home Assistant calendars."""
    
    name = "calendar"
    description = "Create calendar events on connected calendars."
    
    # Field definitions
    # Note: datetime is special - either start_date OR start_date_time is required
    MANDATORY_FIELDS = ["summary", "datetime", "calendar_id"]
    OPTIONAL_FIELDS = ["description", "location", "duration_minutes"]
    
    FIELD_PROMPTS = {
        "summary": "Wie soll der Termin heißen?",
        "datetime": "Wann soll der Termin sein?",
        "calendar_id": "In welchen Kalender?",  # Will be customized with calendar list
    }
    
    # Prompt to extract calendar event details from natural language
    PROMPT = {
        "system": """Extract calendar event details from the user's request.

Parse the following information if present:
- summary: Event title/name (required)
- description: Additional details about the event
- start_date: Start date in YYYY-MM-DD format (for all-day events)
- end_date: End date in YYYY-MM-DD format (for all-day events, day AFTER the event ends)
- start_date_time: Start date and time in YYYY-MM-DD HH:MM format (for timed events)
- end_date_time: End date and time in YYYY-MM-DD HH:MM format (for timed events)
- location: Event location
- duration_minutes: Duration in minutes if no end time is specified
- is_all_day: true if no specific time is mentioned

Today's date for reference: {today}

Examples:
"Termin morgen um 10 Uhr beim Zahnarzt" → {{"summary": "Zahnarzt", "start_date_time": "2023-12-14 10:00", "duration_minutes": 60}}
"Geburtstag am 25. Dezember ganztägig" → {{"summary": "Geburtstag", "start_date": "2023-12-25", "end_date": "2023-12-26", "is_all_day": true}}
"Meeting in 2 Stunden" → {{"summary": "Meeting", "start_date_time": "2023-12-13 14:00", "duration_minutes": 60}}
"Arzttermin nächsten Montag 14:30 in der Praxis Dr. Müller" → {{"summary": "Arzttermin", "start_date_time": "2023-12-18 14:30", "location": "Praxis Dr. Müller", "duration_minutes": 60}}
""",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "start_date_time": {"type": "string"},
                "end_date_time": {"type": "string"},
                "location": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "is_all_day": {"type": "boolean"},
            },
        },
    }
    
    # Store calendar list for selection
    _calendars: List[Dict[str, str]] = []
    
    async def run(
        self, user_input, intent_name: str = None, slots: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Handle calendar intent."""
        slots = slots or {}
        
        # Accept calendar-related intents
        if intent_name and intent_name not in ("HassCalendarCreate", "HassCreateEvent", "HassCalendarAdd"):
            return {}
        
        # Extract event details from natural language using LLM
        event_data = await self._extract_event_details(user_input.text)
        if not event_data:
            event_data = {}
        
        # Merge with slots from NLU
        if slots.get("summary"):
            event_data["summary"] = slots["summary"]
        if slots.get("location"):
            event_data["location"] = slots["location"]
        
        # Only use calendar slot if it's a valid entity_id
        slot_calendar = slots.get("calendar", "")
        if slot_calendar and slot_calendar.startswith("calendar."):
            if self.hass.states.get(slot_calendar):
                event_data["calendar_id"] = slot_calendar
        
        # Handle date/time combination from slots
        slot_date = slots.get("date", "")
        slot_time = slots.get("time", "")
        slot_duration = slots.get("duration", "")
        
        if slot_date and slot_time:
            time_match = re.search(r'(\d{1,2})(?:[:.](\d{2}))?\s*[Uu]hr', slot_time)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                event_data["start_date_time"] = f"{slot_date} {hour:02d}:{minute:02d}"
                
                end_match = re.search(r'bis\s+(\d{1,2})(?:[:.](\d{2}))?\s*[Uu]hr', slot_time)
                if end_match:
                    end_hour = int(end_match.group(1))
                    end_minute = int(end_match.group(2) or 0)
                    event_data["end_date_time"] = f"{slot_date} {end_hour:02d}:{end_minute:02d}"
            else:
                event_data["start_date"] = slot_date
        elif slot_date and not slot_time:
            event_data["start_date"] = slot_date
        
        if slot_duration and not event_data.get("duration_minutes"):
            duration_minutes = self._parse_duration(slot_duration)
            if duration_minutes:
                event_data["duration_minutes"] = duration_minutes
        
        # Store data for processing
        return await self._process(user_input, event_data)
    
    async def _extract_fields(self, text: str) -> Dict[str, Any]:
        """Extract calendar fields from natural language."""
        return await self._extract_event_details(text) or {}
    
    async def _extract_event_details(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract event details using LLM."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            prompt = dict(self.PROMPT)
            prompt["system"] = prompt["system"].format(today=today)
            
            result = await self._safe_prompt(
                prompt, {"user_input": text}, temperature=0.0
            )
            if result and isinstance(result, dict):
                return result
        except Exception as e:
            _LOGGER.debug(f"Failed to extract event details: {e}")
        return None
    
    def _has_field(self, data: Dict[str, Any], field: str) -> bool:
        """Check if field has a valid value - special handling for datetime."""
        if field == "datetime":
            # Either start_date or start_date_time counts as having datetime
            return bool(data.get("start_date") or data.get("start_date_time"))
        return super()._has_field(data, field)
    
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and resolve data - handle calendars and dates."""
        # Refresh calendar list
        self._calendars = self._get_calendar_entities()
        
        # Auto-select calendar if only one exists
        if not data.get("calendar_id"):
            if len(self._calendars) == 1:
                data["calendar_id"] = self._calendars[0]["entity_id"]
                _LOGGER.debug("[Calendar] Auto-selected single calendar: %s", data["calendar_id"])
        
        # Resolve relative dates
        data = self._resolve_relative_dates(data)
        
        return data
    
    async def _ask_for_field(
        self, user_input, field: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Custom field prompts for calendar."""
        if field == "datetime":
            return {
                "status": "handled",
                "result": await make_response("Wann soll der Termin sein?", user_input),
                "pending_data": {
                    "type": self.name,
                    "step": "ask_datetime",
                    "event_data": data,
                },
            }
        
        if field == "calendar_id":
            if not self._calendars:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Keine Kalender gefunden. Bitte richte zuerst einen Kalender in Home Assistant ein.",
                        user_input,
                    ),
                }
            
            calendar_names = [c["name"] for c in self._calendars]
            return {
                "status": "handled",
                "result": await make_response(
                    f"In welchen Kalender? ({', '.join(calendar_names)})",
                    user_input,
                ),
                "pending_data": {
                    "type": self.name,
                    "step": "ask_calendar",
                    "event_data": data,
                    "calendars": self._calendars,
                },
            }
        
        # Use standard field prompt for summary
        return {
            "status": "handled",
            "result": await make_response(
                self.FIELD_PROMPTS.get(field, f"Bitte gib {field} an."),
                user_input,
            ),
            "pending_data": {
                "type": self.name,
                "step": f"ask_{field}",
                "event_data": data,
            },
        }
    
    async def _process(
        self, user_input, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Custom processing for calendar - maintains existing complex logic."""
        # 1. Validate/transform data
        data = await self._validate_data(data)
        
        # 2. Check for summary (REQUIRED)
        if not data.get("summary"):
            return await self._ask_for_field(user_input, "summary", data)
        
        # 3. Check for date/time (REQUIRED)
        if not self._has_field(data, "datetime"):
            return await self._ask_for_field(user_input, "datetime", data)
        
        # 4. Check for calendar (REQUIRED if multiple)
        if not data.get("calendar_id"):
            return await self._ask_for_field(user_input, "calendar_id", data)
        
        # 5. Validate date formats
        if not self._validate_dates(data):
            _LOGGER.debug("[Calendar] Date validation failed: %s", data)
            return {
                "status": "handled",
                "result": await make_response(
                    "Bitte gib ein konkretes Datum an, z.B. 'am 14. Dezember' oder 'am Montag um 15 Uhr'.",
                    user_input,
                ),
                "pending_data": {
                    "type": self.name,
                    "step": "ask_datetime",
                    "event_data": {k: v for k, v in data.items() 
                                  if k not in ("start_date", "end_date", "start_date_time", "end_date_time")},
                },
            }
        
        # 6. Calculate end time if not specified
        data = self._calculate_end_time(data)
        
        # 7. Show confirmation
        return await self._show_confirmation(user_input, data)
    
    async def _show_confirmation(
        self, user_input, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Show confirmation with event_data key for backwards compatibility."""
        confirmation_text = self._build_confirmation_text(data)
        
        return {
            "status": "handled",
            "result": await make_response(
                f"Termin erstellen?\n{confirmation_text}\n\nSag 'Ja' zum Bestätigen.",
                user_input,
            ),
            "pending_data": {
                "type": self.name,
                "step": "confirm",
                "event_data": data,  # Use event_data for backwards compatibility
            },
        }
    
    async def continue_flow(
        self, user_input, pending_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Continue multi-turn calendar event creation flow."""
        step = pending_data.get("step")
        event_data = pending_data.get("event_data", pending_data.get("data", {}))
        text = user_input.text.strip()
        
        # Restore calendars list
        calendars = pending_data.get("calendars", [])
        if calendars:
            self._calendars = calendars
        
        if step == "ask_summary":
            event_data["summary"] = text
            
        elif step == "ask_datetime":
            parsed = await self._parse_datetime(text)
            if parsed:
                event_data.update(parsed)
            else:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Ich habe das Datum nicht verstanden. Bitte sag z.B. 'morgen um 10 Uhr' oder '25. Dezember'.",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
                
        elif step == "ask_calendar":
            matched = await self._fuzzy_match_calendar(text, self._calendars)
            if not matched:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Das habe ich nicht verstanden. Welcher Kalender?",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
            event_data["calendar_id"] = matched
            
        elif step == "confirm":
            if self._is_affirmative(text):
                return await self._execute(user_input, event_data)
            elif self._is_negative(text):
                return {
                    "status": "handled",
                    "result": await make_response("Termin wurde nicht erstellt.", user_input),
                }
            else:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Sag 'Ja' zum Bestätigen oder 'Nein' zum Abbrechen.",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
        
        # Continue processing with updated data
        return await self._process(user_input, event_data)
    
    async def _parse_datetime(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse date/time from user input using LLM."""
        return await self._extract_event_details(f"Termin {text}")
    
    async def _build_confirmation(self, data: Dict[str, Any]) -> str:
        """Build confirmation text."""
        return self._build_confirmation_text(data)
    
    async def _execute(self, user_input, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create the calendar event."""
        calendar_id = data.get("calendar_id")
        if not calendar_id:
            return {
                "status": "handled",
                "result": await make_response("Fehler: Kein Kalender ausgewählt.", user_input),
            }
        
        # Build service data
        service_data = {
            "summary": data.get("summary", "Termin"),
        }
        
        # Add date/time
        if data.get("start_date_time"):
            service_data["start_date_time"] = data["start_date_time"] + ":00"
            service_data["end_date_time"] = data.get("end_date_time", data["start_date_time"]) + ":00"
        elif data.get("start_date"):
            service_data["start_date"] = data["start_date"]
            service_data["end_date"] = data.get("end_date", data["start_date"])
        
        # Add optional fields
        if data.get("description"):
            service_data["description"] = data["description"]
        if data.get("location"):
            service_data["location"] = data["location"]
        
        try:
            await self.hass.services.async_call(
                "calendar",
                "create_event",
                service_data,
                target={"entity_id": calendar_id},
            )
            
            summary = data.get("summary", "Termin")
            return {
                "status": "handled",
                "result": await make_response(f"✅ Termin '{summary}' wurde erstellt.", user_input),
            }
        except Exception as e:
            _LOGGER.error(f"Failed to create calendar event: {e}")
            return {
                "status": "handled",
                "result": await make_response(f"Fehler beim Erstellen des Termins: {str(e)}", user_input),
            }
    
    # --- Helper methods preserved from original ---
    
    def _calculate_end_time(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate end time if not specified."""
        if not data.get("end_date") and not data.get("end_date_time"):
            if data.get("start_date_time"):
                duration = data.get("duration_minutes", 60)
                try:
                    start = datetime.strptime(data["start_date_time"], "%Y-%m-%d %H:%M")
                    end = start + timedelta(minutes=duration)
                    data["end_date_time"] = end.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass
            elif data.get("start_date"):
                try:
                    start = datetime.strptime(data["start_date"], "%Y-%m-%d")
                    end = start + timedelta(days=1)
                    data["end_date"] = end.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        return data
    
    def _resolve_relative_dates(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative date terms to actual dates."""
        today = datetime.now()
        
        relative_days = [
            ("übermorgen", 2),
            ("morgen", 1),
            ("heute", 0),
        ]
        
        weekdays = {
            "montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
            "freitag": 4, "samstag": 5, "sonntag": 6,
        }
        
        def resolve_date(value: str) -> str:
            """Resolve a single date value to YYYY-MM-DD format."""
            if not value:
                return value
            
            # Already in correct format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                return value
            
            value_lower = value.lower().strip()
            
            # Check relative days
            for term, days_offset in relative_days:
                if term in value_lower:
                    target = today + timedelta(days=days_offset)
                    return target.strftime("%Y-%m-%d")
            
            # Check "in X Tagen" pattern
            match = re.match(r'in\s+(\d+)\s+tag', value_lower)
            if match:
                days = int(match.group(1))
                target = today + timedelta(days=days)
                return target.strftime("%Y-%m-%d")
            
            # Check "X Tage" pattern (without "in")
            match = re.match(r'(\d+)\s+tag', value_lower)
            if match:
                days = int(match.group(1))
                target = today + timedelta(days=days)
                return target.strftime("%Y-%m-%d")
            
            # Check weekdays
            for day_name, day_num in weekdays.items():
                if day_name in value_lower or f"nächsten {day_name}" in value_lower:
                    days_ahead = day_num - today.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    target = today + timedelta(days=days_ahead)
                    return target.strftime("%Y-%m-%d")
            
            return value
        
        def resolve_datetime(value: str) -> str:
            """Resolve a datetime value, preserving time if present."""
            if not value:
                return value
            
            if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', value):
                return value
            
            parts = value.split(' ')
            if len(parts) >= 2:
                time_part = parts[-1]
                date_part = ' '.join(parts[:-1])
                
                if re.match(r'\d{1,2}:\d{2}', time_part):
                    resolved_date = resolve_date(date_part)
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', resolved_date):
                        if len(time_part) == 4:
                            time_part = "0" + time_part
                        return f"{resolved_date} {time_part}"
            
            resolved = resolve_date(value)
            if resolved != value:
                return f"{resolved} 12:00"
            
            return value
        
        # Resolve start_date
        if event_data.get("start_date"):
            event_data["start_date"] = resolve_date(event_data["start_date"])
        
        # Resolve end_date
        if event_data.get("end_date"):
            event_data["end_date"] = resolve_date(event_data["end_date"])
        
        # Resolve start_date_time
        if event_data.get("start_date_time"):
            event_data["start_date_time"] = resolve_datetime(event_data["start_date_time"])
        
        # Resolve end_date_time
        if event_data.get("end_date_time"):
            event_data["end_date_time"] = resolve_datetime(event_data["end_date_time"])
        
        return event_data
    
    def _parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse duration string to minutes."""
        if not duration_str:
            return None
        
        total_minutes = 0
        text = duration_str.lower()
        
        hours_match = re.search(r'(\d+(?:[,\.]\d+)?)\s*(?:stunde|stunden|std|h)', text)
        if hours_match:
            hours = float(hours_match.group(1).replace(',', '.'))
            total_minutes += int(hours * 60)
        
        minutes_match = re.search(r'(\d+)\s*(?:minute|minuten|min|m)', text)
        if minutes_match:
            total_minutes += int(minutes_match.group(1))
        
        return total_minutes if total_minutes > 0 else None
    
    def _validate_dates(self, event_data: Dict[str, Any]) -> bool:
        """Validate that date fields are in parseable format."""
        if event_data.get("start_date_time"):
            try:
                datetime.strptime(event_data["start_date_time"], "%Y-%m-%d %H:%M")
            except ValueError:
                return False
        
        if event_data.get("end_date_time"):
            try:
                datetime.strptime(event_data["end_date_time"], "%Y-%m-%d %H:%M")
            except ValueError:
                return False
        
        if event_data.get("start_date"):
            try:
                datetime.strptime(event_data["start_date"], "%Y-%m-%d")
            except ValueError:
                return False
        
        if event_data.get("end_date"):
            try:
                datetime.strptime(event_data["end_date"], "%Y-%m-%d")
            except ValueError:
                return False
        
        return True
    
    def _build_confirmation_text(self, event_data: Dict[str, Any]) -> str:
        """Build a human-readable confirmation text."""
        summary = event_data.get("summary", "Termin")
        lines = [f"📅 **{summary}**"]
        
        if event_data.get("start_date_time"):
            try:
                dt = datetime.strptime(event_data["start_date_time"], "%Y-%m-%d %H:%M")
                date_str = dt.strftime("%d.%m.%Y")
                time_str = dt.strftime("%H:%M")
                lines.append(f"🕐 {date_str} um {time_str} Uhr")
            except ValueError:
                lines.append(f"🕐 {event_data['start_date_time']}")
        elif event_data.get("start_date"):
            try:
                dt = datetime.strptime(event_data["start_date"], "%Y-%m-%d")
                date_str = dt.strftime("%d.%m.%Y")
                lines.append(f"📆 {date_str} (ganztägig)")
            except ValueError:
                lines.append(f"📆 {event_data['start_date']}")
        
        if event_data.get("location"):
            lines.append(f"📍 {event_data['location']}")
        
        if event_data.get("calendar_id"):
            calendar_name = event_data["calendar_id"].replace("calendar.", "").replace("_", " ").title()
            for cal in self._calendars:
                if cal["entity_id"] == event_data["calendar_id"]:
                    calendar_name = cal["name"]
                    break
            lines.append(f"📁 Kalender: {calendar_name}")
        
        return "\n".join(lines)
    
    def _get_calendar_entities(self) -> List[Dict[str, str]]:
        """Get all calendar entities exposed to the conversation/assist integration."""
        calendars = []
        
        # Get calendar entity IDs
        try:
            entity_ids = self.hass.states.async_entity_ids("calendar")
        except Exception:
            # Fallback for different mock setups
            entity_ids = []
            try:
                for state in self.hass.states.async_all("calendar"):
                    entity_ids.append(state.entity_id)
            except Exception:
                pass
        
        # Check exposure (gracefully handle test mocks)
        try:
            from homeassistant.components.conversation import async_should_expose
            use_exposure_check = True
        except ImportError:
            use_exposure_check = False
        
        for entity_id in entity_ids:
            # Check exposure if available
            if use_exposure_check:
                try:
                    if not async_should_expose(self.hass, "conversation", entity_id):
                        continue
                except Exception:
                    pass  # Skip exposure check if it fails
            
            # Get state and attributes
            state = self.hass.states.get(entity_id)
            if state:
                name = state.attributes.get("friendly_name", entity_id.split(".")[-1])
                calendars.append({
                    "entity_id": entity_id,
                    "name": name,
                })
        
        return calendars
    
    async def _fuzzy_match_calendar(
        self, query: str, calendars: List[Dict[str, str]]
    ) -> Optional[str]:
        """Match user input to a calendar using fuzzy matching."""
        # Use centralized fuzzy matching utility
        return await fuzzy_match_candidates(
            query,
            calendars,
            name_key="name",
            id_key="entity_id",
            threshold=60,
        )
