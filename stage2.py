import logging
from typing import Any, Dict, List
from .base_stage import BaseStage
from .capabilities.disambiguation_resolution import DisambiguationResolutionCapability
from .capabilities.plural_detection import PluralDetectionCapability
from .capabilities.intent_executor import IntentExecutorCapability
from .conversation_utils import make_response, error_response
from .stage_result import Stage0Result

_LOGGER = logging.getLogger(__name__)


class Stage2Processor(BaseStage):
    name = "stage2"
    capabilities = [
        DisambiguationResolutionCapability,
        PluralDetectionCapability,
        IntentExecutorCapability,
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage2] Input='%s'", user_input.text)

        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        if key in self._pending:
            _LOGGER.debug("[Stage2] Handling pending disambiguation follow-up.")
            pending = self._pending.pop(key)
            candidates = [{"entity_id": k, "name": v} for k, v in pending.get("candidates", {}).items()]
            result = await self.use("disambiguation_resolution", user_input, candidates=candidates)
            if not result or not result.get("entities"):
                return {"status": "error", "result": await error_response(user_input, "Ich habe das nicht verstanden.")}
            return {"status": "handled", "result": await make_response(result.get("message"), user_input)}

        # If we have a Stage0Result with multiple entities, consider plural execution here as a fallback
        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            pd = await self.use("plural_detection", user_input) or {}
            if pd.get("multiple_entities") is True and prev_result.intent:
                _LOGGER.debug("[Stage2] Detected plural entities → executing collective intent via IntentExecutorCapability.")
                exec_data = await self.use(
                    "intent_executor",
                    user_input,
                    intent_name=prev_result.intent,
                    entity_ids=list(prev_result.resolved_ids),
                    language=user_input.language or "de",
                )
                if exec_data and exec_data.get("result"):
                    return {"status": "handled", "result": exec_data["result"]}
                _LOGGER.warning("[Stage2] IntentExecutorCapability returned no result for plural execution.")
                return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}

            _LOGGER.debug("[Stage2] <=1 resolved entity or plural not confirmed → nothing to do here, escalate.")
            return {"status": "escalate", "result": prev_result}

        _LOGGER.debug("[Stage2] No special handling → marking as error to trigger fallback.")
        return {"status": "error", "result": None}
