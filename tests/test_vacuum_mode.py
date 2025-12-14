"""Tests for vacuum mode detection in VacuumCapability."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from homeassistant.components import conversation

from multistage_assist.capabilities.vacuum import VacuumCapability
from tests.integration import get_llm_config


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


@pytest.fixture
def vacuum_capability(hass):
    """Create vacuum capability with real LLM."""
    return VacuumCapability(hass, get_llm_config())


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
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_mode", [
        ("Sauge den Keller", "vacuum"),
        ("Staubsauge das Wohnzimmer", "vacuum"),
        ("Saugen im Büro", "vacuum"),
        ("Sauge die Küche", "vacuum"),
    ])
    async def test_vacuum_mode_detection(self, vacuum_capability, input_text, expected_mode):
        """Test that vacuum mode is correctly detected from German text."""
        extracted = await vacuum_capability._extract_vacuum_details(input_text)
        
        actual_mode = extracted.get("mode", "vacuum")
        assert actual_mode == expected_mode, \
            f"For '{input_text}': expected mode='{expected_mode}', got mode='{actual_mode}'"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text", [
        "Wische die Küche",
        "Wischen im Bad",
        "wische den Keller",
    ])
    async def test_mop_commands(self, vacuum_capability, input_text):
        """Test that mop commands have mode='mop'."""
        extracted = await vacuum_capability._extract_vacuum_details(input_text)
        
        assert extracted.get("mode") == "mop", \
            f"For '{input_text}': expected mode='mop', got mode='{extracted.get('mode')}'"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_area", [
        ("Sauge den Keller", "Keller"),
        ("Staubsauge das Wohnzimmer", "Wohnzimmer"),
        ("Wische die Küche", "Küche"),
        ("wische den Keller", "Keller"),
    ])
    async def test_vacuum_area_extraction(self, vacuum_capability, input_text, expected_area):
        """Test that area is correctly extracted from vacuum commands."""
        extracted = await vacuum_capability._extract_vacuum_details(input_text)
        
        actual_area = extracted.get("area", "")
        
        # Case-insensitive comparison
        assert actual_area.lower() == expected_area.lower(), \
            f"For '{input_text}': expected area='{expected_area}', got area='{actual_area}'"


class TestVacuumFallback:
    """Test that vacuum defaults work when LLM fails."""
    
    @pytest.mark.asyncio
    async def test_mode_defaults_to_vacuum(self, hass):
        """Test that mode defaults to 'vacuum' if extraction fails."""
        cap = VacuumCapability(hass, {})
        
        # Mock _safe_prompt to return empty dict (LLM failure)
        cap._safe_prompt = AsyncMock(return_value={})
        
        result = await cap._extract_vacuum_details("Sauge irgendwas")
        
        # Should return empty dict, but mode in run() defaults to "vacuum"
        assert result == {}
