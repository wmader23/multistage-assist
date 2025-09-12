"""Config flow for Multi-Stage Assist."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback

from .const import (
    DOMAIN,
    CONF_STAGE1_IP,
    CONF_STAGE1_PORT,
    CONF_STAGE1_MODEL,
    CONF_STAGE2_IP,
    CONF_STAGE2_PORT,
    CONF_STAGE2_MODEL,
)


class MultiStageAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle config flow."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """UI step for initial setup."""
        errors = {}

        if user_input is not None:
            return self.async_create_entry(title="Multi-Stage Assist", data=user_input)

        schema = vol.Schema(
            {
                vol.Optional(CONF_STAGE1_IP, default="127.0.0.1"): str,
                vol.Optional(CONF_STAGE1_PORT, default=5000): int,
                vol.Optional(CONF_STAGE1_MODEL, default="entity_filter_model"): str,
                vol.Optional(CONF_STAGE2_IP, default="127.0.0.1"): str,
                vol.Optional(CONF_STAGE2_PORT, default=5001): int,
                vol.Optional(CONF_STAGE2_MODEL, default="llm_model"): str,
            }
        )

        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return MultiStageAssistOptionsFlowHandler(config_entry)


class MultiStageAssistOptionsFlowHandler(config_entries.OptionsFlow):
    """Options flow for editing config."""

    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        data = self.config_entry.data

        schema = vol.Schema(
            {
                vol.Optional(CONF_STAGE1_IP, default=data.get(CONF_STAGE1_IP, "127.0.0.1")): str,
                vol.Optional(CONF_STAGE1_PORT, default=data.get(CONF_STAGE1_PORT, 5000)): int,
                vol.Optional(CONF_STAGE1_MODEL, default=data.get(CONF_STAGE1_MODEL, "entity_filter_model")): str,
                vol.Optional(CONF_STAGE2_IP, default=data.get(CONF_STAGE2_IP, "127.0.0.1")): str,
                vol.Optional(CONF_STAGE2_PORT, default=data.get(CONF_STAGE2_PORT, 5001)): int,
                vol.Optional(CONF_STAGE2_MODEL, default=data.get(CONF_STAGE2_MODEL, "llm_model")): str,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
