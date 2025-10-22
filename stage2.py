import ast
import logging
from typing import Any, Dict, List, Callable

from homeassistant.components import conversation
from homeassistant.helpers import intent

from .prompt_executor import PromptExecutor
from .entity_resolver import EntityResolver
from .prompts import (
    DISAMBIGUATION_RESOLUTION_PROMPT,
    GET_VALUE_PHRASE_PROMPT,
    SENSOR_SELECTION_PROMPT,
)
from .stage0 import Stage0Result
from .conversation_utils import make_response, abort_response, error_response

_LOGGER = logging.getLogger(__name__)


class SafeLambdaError(Exception):
    pass


def _safe_compile_lambda(src: str) -> Callable[[List[float]], float]:
    """Compile and validate restricted lambda safely."""
    if not src or "lambda" not in src:
        raise SafeLambdaError("Function must be a lambda.")
    try:
        tree = ast.parse(src, mode="eval")
    except Exception as e:
        raise SafeLambdaError(f"Parse error: {e}") from e

    allowed_calls = {"sum", "len", "max", "min", "abs", "round"}
    allowed_names = {"values"}

    class Validator(ast.NodeVisitor):
        def visit_Lambda(self, node: ast.Lambda):
            if not isinstance(node.args, ast.arguments) or len(node.args.args) != 1:
                raise SafeLambdaError("Lambda must have exactly one parameter 'values'.")
            if node.args.args[0].arg != "values":
                raise SafeLambdaError("Lambda parameter must be named 'values'.")
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            if node.id not in allowed_names and node.id not in allowed_calls:
                raise SafeLambdaError(f"Name '{node.id}' not allowed.")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                raise SafeLambdaError("Only builtins sum,len,max,min,abs,round are allowed.")
            self.generic_visit(node)

        def generic_visit(self, node):
            forbidden = (
                ast.Attribute, ast.Dict, ast.DictComp, ast.ListComp, ast.SetComp,
                ast.GeneratorExp, ast.Subscript, ast.IfExp, ast.Compare,
                ast.BoolOp, ast.And, ast.Or
            )
            if isinstance(node, forbidden):
                raise SafeLambdaError(f"Forbidden node: {type(node).__name__}")
            super().generic_visit(node)

    Validator().visit(tree)
    code = compile(tree, "<safe_lambda>", "eval")
    func = eval(code, {"__builtins__": {}},
                {"sum": sum, "len": len, "max": max, "min": min, "abs": abs, "round": round})
    if not callable(func):
        raise SafeLambdaError("Compiled object is not callable.")
    return func  # type: ignore[return-value]


class Stage2Processor:
    """Stage 2: Execute intents, handle pending disambiguation, and compute sensor values."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)
        self.entities = EntityResolver(hass)
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _get_state_key(self, user_input) -> str:
        return getattr(user_input, "session_id", None) or user_input.conversation_id

    def has_pending(self, user_input) -> bool:
        return self._get_state_key(user_input) in self._pending

    async def resolve_pending(self, user_input):
        key = self._get_state_key(user_input)
        pending = self._pending.get(key)
        _LOGGER.debug(
            "Resolving pending for %s (%s candidates)",
            key,
            len(pending.get("candidates", {})) if pending else 0,
        )

        if not pending:
            return await conversation.async_converse(
                self.hass,
                text=user_input.text,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=user_input.language or "de",
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )

        candidates_ordered = [
            {"entity_id": eid, "name": name}
            for eid, name in pending["candidates"].items()
        ]

        data = await self.prompts.run(
            DISAMBIGUATION_RESOLUTION_PROMPT,
            {
                "user_input": user_input.text,
                "input_entities": candidates_ordered,
            },
            temperature=0.25,
        )
        _LOGGER.debug("Resolution output: %s", data)

        entities = (data or {}).get("entities") or []
        message = (data or {}).get("message") or ""
        action = (data or {}).get("action")

        if action == "abort":
            self._pending.pop(key, None)
            return await abort_response(user_input)

        if not entities:
            return await error_response(user_input)

        if pending["kind"] == "action":
            return await self._resolve_action(user_input, pending, entities, message)

        if pending["kind"] == "value":
            return await self._resolve_value(user_input, pending, entities)

        self._pending.pop(key, None)
        return await error_response(user_input, "Entschuldigung, das konnte ich nicht ausführen.")

    async def _resolve_action(self, user_input, pending, entities: List[str], message: str):
        intent_obj = pending.get("intent")
        self._pending.pop(self._get_state_key(user_input), None)
        if not intent_obj:
            return await error_response(user_input, "Entschuldigung, das konnte ich nicht ausführen.")

        try:
            intent_name = getattr(intent_obj, "name", None)
            last_resp = None
            for eid in entities:
                slots: Dict[str, Any] = {"name": {"value": eid}}
                _LOGGER.debug("Executing intent '%s' for %s", intent_name, eid)
                last_resp = await intent.async_handle(
                    self.hass,
                    platform="conversation",
                    intent_type=intent_name,
                    slots=slots,
                    text_input=user_input.text,
                    context=user_input.context,
                    language=user_input.language or "de",
                )

            if last_resp and not last_resp.speech:
                if message:
                    last_resp.async_set_speech(message)
                else:
                    pretty = ", ".join(pending["candidates"].get(eid, eid) for eid in entities)
                    last_resp.async_set_speech(f"Okay, {pretty} ist erledigt.")

            return conversation.ConversationResult(
                response=last_resp,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )
        except Exception as e:
            _LOGGER.exception("Failed to execute disambiguated action: %s", e)
            return await error_response(user_input, "Entschuldigung, das konnte ich nicht ausführen.")

    async def _resolve_value(self, user_input, pending, entities: List[str]):
        self._pending.pop(self._get_state_key(user_input), None)
        clarification_data = pending.get("clarification_data") or {}

        if len(entities) == 1:
            return await self._handle_get_value(user_input, entities[0], clarification_data)

        return await self._run_sensor_selection(user_input, entities, clarification_data)

    async def _handle_get_value(self, user_input, eid: str, data: Dict[str, Any]):
        state = self.hass.states.get(eid)
        if not state or state.state in ("unknown", "unavailable", None):
            return await error_response(user_input, "Der Wert ist derzeit nicht verfügbar.")

        context = {
            "measurement": data.get("measurement") or "Wert",
            "value": state.state,
            "unit": state.attributes.get("unit_of_measurement") or "",
            "area": data.get("area"),
        }
        phrased = await self.prompts.run(GET_VALUE_PHRASE_PROMPT, context)
        message = (phrased or {}).get("message") or f"{context['measurement']}: {context['value']} {context['unit']}"
        return await make_response(message, user_input, end=True)

    async def _run_sensor_selection(self, user_input, sensors: List[str], clarification_data: Dict[str, Any]):
        selection = await self.prompts.run(
            SENSOR_SELECTION_PROMPT,
            {
                "user_input": user_input.text,
                "measurement": clarification_data.get("measurement"),
            },
        )
        _LOGGER.debug("SENSOR_SELECTION_PROMPT output: %s", selection)

        chosen = [e for e in (selection or {}).get("entities", []) if isinstance(e, str)]
        func_src = (selection or {}).get("function")

        if chosen:
            chosen = [e for e in chosen if e in sensors]
        else:
            chosen = list(sensors)

        if not chosen:
            return await error_response(user_input, "Ich konnte keinen passenden Sensor finden.")

        if len(chosen) == 1 and not func_src:
            return await self._handle_get_value(user_input, chosen[0], clarification_data)

        values: List[float] = []
        units: List[str] = []
        for eid in chosen:
            st = self.hass.states.get(eid)
            if not st:
                continue
            try:
                val = float(str(st.state).replace(",", "."))
            except Exception:
                continue
            values.append(val)
            unit = st.attributes.get("unit_of_measurement")
            if unit:
                units.append(str(unit))

        if not values:
            return await error_response(user_input, "Die Werte sind derzeit nicht verfügbar.")

        try:
            func = _safe_compile_lambda(func_src or "lambda values: sum(values)/len(values)")
            result_value = func(values)
        except Exception as e:
            _LOGGER.exception("Aggregation function failed: %s", e)
            result_value = sum(values) / len(values)

        unit_out = units[0] if units else ""
        context = {
            "measurement": clarification_data.get("measurement") or "Wert",
            "value": result_value,
            "unit": unit_out,
            "area": clarification_data.get("area"),
        }
        phrased = await self.prompts.run(GET_VALUE_PHRASE_PROMPT, context)
        message = (phrased or {}).get("message") or f"{context['measurement']}: {result_value} {unit_out}"
        return await make_response(message, user_input, end=True)

    async def run(self, user_input: conversation.ConversationInput, s0: Stage0Result):
        """Execute resolved intents or continue pending disambiguations."""
        merged_ids = list(s0.resolved_ids or [])
        _LOGGER.debug("Stage2 received resolved ids: %s", merged_ids)

        intent_name = getattr(s0.intent, "name", None)
        if intent_name == "HassGetState":
            sensors = [eid for eid in merged_ids if eid.startswith("sensor.") and self.hass.states.get(eid) is not None]
            clarification_data = {"measurement": None, "area": None}

            if len(sensors) == 1:
                return await self._handle_get_value(user_input, sensors[0], clarification_data)
            if len(sensors) > 1:
                return await self._run_sensor_selection(user_input, sensors, clarification_data)
            if not sensors and merged_ids:
                # If sensors ambiguous, handle via value disambiguation prompt
                return await self._resolve_value(user_input, {"clarification_data": clarification_data}, merged_ids)
            return await error_response(user_input, "Ich konnte keinen passenden Sensor finden.")

        if len(merged_ids) == 0:
            return await error_response(user_input, "Ich konnte kein passendes Gerät finden.")

        # Stage1 now handles disambiguation → no branching here
        if len(merged_ids) == 1:
            return await conversation.async_converse(
                self.hass,
                text=user_input.text,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=user_input.language or "de",
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )

        # Fallback safeguard only (should not be reached normally)
        _LOGGER.warning("Stage2 received multiple IDs unexpectedly (disambiguation should happen in Stage1)")
        return await error_response(user_input, "Bitte präzisiere, welches Gerät du meinst.")
