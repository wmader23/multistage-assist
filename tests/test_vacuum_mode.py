"""Tests for vacuum mode detection in keyword_intent."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
from tests.integration import get_llm_config


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def keyword_intent_capability(hass):
    """Create keyword intent capability with real LLM."""
    return KeywordIntentCapability(hass, get_llm_config())


def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )


class TestVacuumModeDetection:
    """Tests for vacuum mode detection (vacuum vs mop)."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_mode", [
        # Vacuum mode (saugen, staubsaugen, etc.) - MUST be vacuum
        ("Sauge den Keller", "vacuum"),
        ("Staubsauge das Wohnzimmer", "vacuum"),
        ("Saugen im Büro", "vacuum"),
        ("Staubsauger starten", "vacuum"),
        ("Sauge die Küche", "vacuum"),
    ])
    async def test_vacuum_mode_detection(self, keyword_intent_capability, input_text, expected_mode):
        """Test that vacuum mode is correctly detected from German text."""
        user_input = make_input(input_text)
        
        result = await keyword_intent_capability.run(user_input)
        
        assert result.get("domain") == "vacuum", f"Expected domain='vacuum', got: {result.get('domain')}"
        assert result.get("intent") == "HassVacuumStart", f"Expected intent='HassVacuumStart', got: {result.get('intent')}"
        
        slots = result.get("slots", {})
        actual_mode = slots.get("mode", "")
        
        assert actual_mode == expected_mode, \
            f"For '{input_text}': expected mode='{expected_mode}', got mode='{actual_mode}'"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text", [
        # Mop commands should trigger vacuum domain with mode='mop'
        "Wische die Küche",
        "Wischen im Bad",
        "wische den Keller",
    ])
    async def test_mop_commands_trigger_vacuum_domain(self, keyword_intent_capability, input_text):
        """Test that mop commands trigger the vacuum domain with mode='mop'."""
        user_input = make_input(input_text)
        
        result = await keyword_intent_capability.run(user_input)
        
        # Should at least detect vacuum domain
        assert result.get("domain") == "vacuum", \
            f"For '{input_text}': expected domain='vacuum', got: {result.get('domain')}"
        assert result.get("intent") == "HassVacuumStart", \
            f"For '{input_text}': expected intent='HassVacuumStart', got: {result.get('intent')}"
        
        # Mode should be 'mop' for wischen commands
        slots = result.get("slots", {})
        assert slots.get("mode") == "mop", \
            f"For '{input_text}': expected mode='mop', got mode='{slots.get('mode')}'"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_area", [
        ("Sauge den Keller", "Keller"),
        ("Staubsauge das Wohnzimmer", "Wohnzimmer"),
        ("Wische die Küche", "Küche"),
        ("wische den Keller", "Keller"),  # Test article stripping
    ])
    async def test_vacuum_area_extraction(self, keyword_intent_capability, input_text, expected_area):
        """Test that area is correctly extracted from vacuum commands (without articles)."""
        user_input = make_input(input_text)
        
        result = await keyword_intent_capability.run(user_input)
        
        slots = result.get("slots", {})
        actual_area = slots.get("area", "")
        
        # Case-insensitive comparison
        assert actual_area.lower() == expected_area.lower(), \
            f"For '{input_text}': expected area='{expected_area}', got area='{actual_area}'"
