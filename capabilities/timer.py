import logging
import re
import asyncio
from typing import Any, Dict, List, Optional
import importlib

from .base import Capability
from custom_components.multistage_assist.conversation_utils import make_response, parse_duration_string, format_seconds_to_string

_LOGGER = logging.getLogger(__name__)

_fuzz = None
async def _get_fuzz():
    global _fuzz
    if _fuzz is not None: return _fuzz
    loop = asyncio.get_running_loop()
    def _load(): return importlib.import_module("rapidfuzz.fuzz")
    _fuzz = await loop.run_in_executor(None, _load)
    return _fuzz

class TimerCapability(Capability):
    name = "timer"
    description = "Set timers on mobile devices."

    async def run(self, user_input, intent_name: str, slots: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        if intent_name != "HassTimerSet": return {}
        duration_raw = slots.get("duration")
        device_name = slots.get("name")
        device_id = slots.get("device_id") 
        return await self._process_request(user_input, duration_raw, device_name, device_id)

    async def continue_flow(self, user_input, pending_data: Dict[str, Any]) -> Dict[str, Any]:
        step = pending_data.get("step")
        duration = pending_data.get("duration")
        device_id = pending_data.get("device_id")
        original_name = pending_data.get("original_name_query")
        
        text = user_input.text
        learning_data = None

        if step == "ask_duration":
            seconds = parse_duration_string(text)
            if not seconds:
                return {"status": "handled", "result": await make_response("Zeit nicht verstanden. Bsp: '5 Minuten'.", user_input), "pending_data": pending_data}
            duration = seconds

        elif step == "ask_device":
            candidates = pending_data.get("candidates", [])
            matched = await self._fuzzy_match_device(text, candidates)
            if not matched:
                return {"status": "handled", "result": await make_response("Das habe ich nicht verstanden. Welches Gerät?", user_input), "pending_data": pending_data}
            device_id = matched
            if original_name:
                learning_data = {"type": "entity", "source": original_name, "target": device_id}

        res = await self._process_request(user_input, duration, device_name=original_name, device_id=device_id)
        if learning_data: res["learning_data"] = learning_data
        return res

    async def _process_request(self, user_input, duration_raw, device_name=None, device_id=None) -> Dict[str, Any]:
        seconds = parse_duration_string(duration_raw) if duration_raw else 0
        
        if not seconds:
            return {"status": "handled", "result": await make_response("Wie lange?", user_input), "pending_data": {"type": "timer", "step": "ask_duration", "device_id": device_id, "name": device_name}}

        if not device_id:
            mobile_services = self._get_mobile_notify_services()
            if not mobile_services:
                return {"status": "handled", "result": await make_response("Keine Geräte gefunden.", user_input)}
            
            if device_name:
                device_id = await self._fuzzy_match_device(device_name, mobile_services)
            
            if not device_id:
                if len(mobile_services) == 1:
                    device_id = mobile_services[0]["service"]
                else:
                    names = [d["name"] for d in mobile_services]
                    return {"status": "handled", "result": await make_response(f"Auf welchem Gerät? ({', '.join(names)})", user_input), "pending_data": {"type": "timer", "step": "ask_device", "duration": seconds, "candidates": mobile_services, "original_name_query": device_name}}

        await self._set_android_timer(device_id, seconds)
        
        # PREFER FRIENDLY NAME FOR RESPONSE
        device_friendly = device_name # Start with what user said
        
        # Try to find official name from service list to be sure
        services = self._get_mobile_notify_services()
        official_name = next((s["name"] for s in services if s["service"] == device_id), None)
        
        # If we have an official name, use it (it's cleaner than raw ID)
        # If we have a user provided name (e.g. "Daniel's Handy") and it matched, maybe use that?
        # A mix is good: "Timer auf Daniel's Handy (SM A566B) gestellt" or just "Timer auf Daniel's Handy gestellt".
        # Let's use the official name if available, or fallback to user query.
        
        if official_name:
            device_friendly = official_name
        elif not device_friendly:
             device_friendly = device_id.split(".")[-1].replace("mobile_app_", "").replace("_", " ").title()

        return {"status": "handled", "result": await make_response(f"Timer für {format_seconds_to_string(seconds)} auf {device_friendly} gestellt.", user_input)}

    def _get_mobile_notify_services(self) -> List[Dict[str, str]]:
        services = self.hass.services.async_services().get("notify", {})
        return [{"service": f"notify.{k}", "name": k.replace("mobile_app_", "").replace("_", " ").title()} for k in services if k.startswith("mobile_app_")]

    async def _fuzzy_match_device(self, query: str, candidates: List[Dict[str, str]]) -> Optional[str]:
        if not query: return None
        fuzz = await _get_fuzz()
        best_score = 0
        best_id = None
        q = query.lower().strip()
        for c in candidates:
            name_score = fuzz.token_set_ratio(q, c["name"].lower())
            id_score = fuzz.token_set_ratio(q, c["service"].split(".")[-1])
            score = max(name_score, id_score)
            if score > best_score and score > 70:
                best_score = score
                best_id = c["service"]
        return best_id

    async def _set_android_timer(self, service_full: str, seconds: int):
        domain, service = service_full.split(".", 1)
        payload = {"message": "command_activity", "data": {"intent_action": "android.intent.action.SET_TIMER", "intent_extras": f"android.intent.extra.alarm.LENGTH:{seconds},android.intent.extra.alarm.SKIP_UI:true"}}
        await self.hass.services.async_call(domain, service, payload)