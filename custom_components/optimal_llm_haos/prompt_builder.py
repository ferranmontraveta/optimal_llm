"""
Prompt Builder for optimal_llm_haos.

Builds the base (cached) and volatile (fresh) prompts from
Home Assistant device and area registries.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

from .const import DEFAULT_CONTEXT_PROMPT, DEFAULT_SYSTEM_PROMPT, LOGGER


class PromptBuilder:
    """Builds prompts for Ollama from Home Assistant registries."""

    def __init__(
        self,
        hass: HomeAssistant,
        system_prompt_template: str | None = None,
        context_prompt_template: str | None = None,
    ) -> None:
        """
        Initialize the prompt builder.

        Args:
            hass: Home Assistant instance
            system_prompt_template: Custom system prompt (uses default if None)
            context_prompt_template: Custom context prompt (uses default if None)

        """
        self._hass = hass
        self._system_prompt_template = system_prompt_template or DEFAULT_SYSTEM_PROMPT
        self._context_prompt_template = context_prompt_template or DEFAULT_CONTEXT_PROMPT

    @property
    def system_prompt_template(self) -> str:
        """Return the current system prompt template."""
        return self._system_prompt_template

    @system_prompt_template.setter
    def system_prompt_template(self, value: str) -> None:
        """Set the system prompt template."""
        self._system_prompt_template = value or DEFAULT_SYSTEM_PROMPT

    @property
    def context_prompt_template(self) -> str:
        """Return the current context prompt template."""
        return self._context_prompt_template

    @context_prompt_template.setter
    def context_prompt_template(self, value: str) -> None:
        """Set the context prompt template."""
        self._context_prompt_template = value or DEFAULT_CONTEXT_PROMPT

    async def async_build_base_prompt(self) -> str:
        """
        Build the base system prompt with device/area information.

        This prompt is cached and reused across conversations.
        It includes static information that rarely changes.

        Returns:
            The complete system prompt string

        """
        device_summary = await self._async_build_device_summary()

        # Try to format with device_summary placeholder
        try:
            return self._system_prompt_template.format(device_summary=device_summary)
        except KeyError:
            # If template doesn't have {device_summary}, append it
            return f"{self._system_prompt_template}\n\n## Devices\n{device_summary}"

    async def _async_build_device_summary(self) -> str:
        """
        Build a summary of all devices grouped by area.

        Returns:
            Formatted string of devices by area

        """
        area_registry = ar.async_get(self._hass)
        device_registry = dr.async_get(self._hass)
        entity_registry = er.async_get(self._hass)

        # Group devices by area
        areas_devices: dict[str, list[dict[str, Any]]] = {}
        unassigned_devices: list[dict[str, Any]] = []

        for device in device_registry.devices.values():
            device_info = {
                "id": device.id,
                "name": device.name_by_user or device.name or "Unknown",
                "manufacturer": device.manufacturer,
                "model": device.model,
                "entities": [],
            }

            # Get entities for this device
            for entity in entity_registry.entities.values():
                if entity.device_id == device.id:
                    device_info["entities"].append({
                        "entity_id": entity.entity_id,
                        "name": entity.name or entity.original_name,
                        "domain": entity.domain,
                    })

            # Group by area
            if device.area_id:
                area = area_registry.async_get_area(device.area_id)
                area_name = area.name if area else "Unknown Area"
                if area_name not in areas_devices:
                    areas_devices[area_name] = []
                areas_devices[area_name].append(device_info)
            else:
                unassigned_devices.append(device_info)

        # Also include entities without devices but assigned to areas
        for entity in entity_registry.entities.values():
            if entity.device_id is None and entity.area_id:
                area = area_registry.async_get_area(entity.area_id)
                area_name = area.name if area else "Unknown Area"
                if area_name not in areas_devices:
                    areas_devices[area_name] = []
                # Create a pseudo-device for the entity
                areas_devices[area_name].append({
                    "id": None,
                    "name": entity.name or entity.original_name or entity.entity_id,
                    "manufacturer": None,
                    "model": None,
                    "entities": [{
                        "entity_id": entity.entity_id,
                        "name": entity.name or entity.original_name,
                        "domain": entity.domain,
                    }],
                })

        # Format the summary
        lines = []
        max_entities_per_device = 5

        for area_name in sorted(areas_devices.keys()):
            devices = areas_devices[area_name]
            lines.append(f"\n### {area_name}")

            for device in devices:
                device_name = device["name"]
                entities = device["entities"]

                if entities:
                    entity_list = ", ".join(
                        f"{e['entity_id']}" for e in entities[:max_entities_per_device]
                    )
                    if len(entities) > max_entities_per_device:
                        extra = len(entities) - max_entities_per_device
                        entity_list += f" (+{extra} more)"
                    lines.append(f"- {device_name}: {entity_list}")
                else:
                    lines.append(f"- {device_name}")

        if unassigned_devices:
            lines.append("\n### Unassigned")
            for device in unassigned_devices:
                device_name = device["name"]
                entities = device["entities"]
                if entities:
                    entity_list = ", ".join(
                        f"{e['entity_id']}" for e in entities[:max_entities_per_device]
                    )
                    lines.append(f"- {device_name}: {entity_list}")
                else:
                    lines.append(f"- {device_name}")

        result = "\n".join(lines) if lines else "No devices registered."
        LOGGER.debug(
            "Built device summary with %d areas and %d unassigned devices",
            len(areas_devices),
            len(unassigned_devices),
        )
        return result

    def build_volatile_context(
        self,
        exposed_entities: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Build the volatile context with current entity states.

        This is regenerated for each conversation turn.

        Args:
            exposed_entities: List of exposed entities with current states

        Returns:
            Formatted string of current entity states

        """
        if not exposed_entities:
            return ""

        lines = []

        for entity in exposed_entities:
            entity_id = entity.get("entity_id", "unknown")
            state = self._hass.states.get(entity_id)

            if state:
                friendly_name = state.attributes.get("friendly_name", entity_id)
                state_str = state.state

                # Add relevant attributes
                attrs = []
                if "brightness" in state.attributes:
                    brightness = state.attributes["brightness"]
                    if brightness:
                        pct = round(brightness / 255 * 100)
                        attrs.append(f"brightness: {pct}%")
                if "temperature" in state.attributes:
                    attrs.append(f"temp: {state.attributes['temperature']}°")
                if "current_temperature" in state.attributes:
                    attrs.append(
                        f"current: {state.attributes['current_temperature']}°"
                    )

                attr_str = f" ({', '.join(attrs)})" if attrs else ""
                lines.append(f"- {friendly_name} ({entity_id}): {state_str}{attr_str}")

        entity_states = "\n".join(lines) if lines else "No entity states available."

        # Format with the context template
        try:
            return self._context_prompt_template.format(entity_states=entity_states)
        except KeyError:
            # If template doesn't have {entity_states}, prepend it
            return f"{entity_states}\n\n{self._context_prompt_template}"

    def build_tools_prompt(self, tools: list[dict[str, Any]]) -> str:
        """
        Build prompt section describing available tools.

        Args:
            tools: List of tool definitions from LLM API

        Returns:
            Formatted string describing available tools

        """
        if not tools:
            return ""

        lines = ["\n## Available Tools\n"]

        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            lines.append(f"- **{name}**: {description}")

        return "\n".join(lines)
