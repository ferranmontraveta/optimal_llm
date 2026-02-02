"""Response templates for optimal_agent - Jinja2 templates for service call responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jinja2 import Template

from .const import LOGGER

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


# Templates keyed by action (matches router's flat "action" field)
# These provide fast, LLM-free responses for common device operations
RESPONSE_TEMPLATES: dict[str, str] = {
    # Lights
    "light.turn_on": (
        "{{ friendly_name }} is now on"
        "{% if brightness is defined %} at {{ (brightness / 255 * 100) | round | int }}%{% endif %}"
        "."
    ),
    "light.turn_off": "{{ friendly_name }} is now off.",
    # Switches
    "switch.turn_on": "{{ friendly_name }} has been turned on.",
    "switch.turn_off": "{{ friendly_name }} has been turned off.",
    # Climate
    "climate.set_temperature": (
        "{{ friendly_name }} set to {{ temperature }}°"
        "{% if hvac_mode is defined %} in {{ hvac_mode }} mode{% endif %}"
        "."
    ),
    "climate.set_hvac_mode": "{{ friendly_name }} set to {{ hvac_mode }} mode.",
    # Covers
    "cover.open_cover": "Opening {{ friendly_name }}.",
    "cover.close_cover": "Closing {{ friendly_name }}.",
    "cover.set_cover_position": "Setting {{ friendly_name }} to {{ position }}%.",
    "cover.stop_cover": "Stopping {{ friendly_name }}.",
    # Fans
    "fan.turn_on": (
        "{{ friendly_name }} is now on"
        "{% if percentage is defined %} at {{ percentage }}%{% endif %}"
        "."
    ),
    "fan.turn_off": "{{ friendly_name }} is now off.",
    "fan.set_percentage": "{{ friendly_name }} set to {{ percentage }}%.",
    # Locks
    "lock.lock": "{{ friendly_name }} is now locked.",
    "lock.unlock": "{{ friendly_name }} is now unlocked.",
    # Scenes and Scripts
    "scene.turn_on": "Activated {{ friendly_name }}.",
    "script.turn_on": "Running {{ friendly_name }}.",
    "script.turn_off": "Stopped {{ friendly_name }}.",
    # Media Players
    "media_player.turn_on": "{{ friendly_name }} is now on.",
    "media_player.turn_off": "{{ friendly_name }} is now off.",
    "media_player.play_media": "Playing on {{ friendly_name }}.",
    "media_player.media_pause": "Paused {{ friendly_name }}.",
    "media_player.media_play": "Resumed {{ friendly_name }}.",
    "media_player.volume_set": "{{ friendly_name }} volume set to {{ (volume_level * 100) | round | int }}%.",
    # Alarm
    "alarm_control_panel.alarm_arm_home": "{{ friendly_name }} armed in home mode.",
    "alarm_control_panel.alarm_arm_away": "{{ friendly_name }} armed in away mode.",
    "alarm_control_panel.alarm_disarm": "{{ friendly_name }} disarmed.",
}

# Fallback template for unknown actions
DEFAULT_TEMPLATE = "Done."

# Error templates
ERROR_TEMPLATES: dict[str, str] = {
    "entity_not_found": "I couldn't find {{ entity_id }}.",
    "service_not_found": "The service {{ action }} is not available.",
    "service_failed": "Failed to execute {{ action }}: {{ error }}.",
    "entity_unavailable": "{{ friendly_name }} is currently unavailable.",
}


def render_response(
    action: str,
    entity_id: str | None,
    params: dict[str, Any],
    hass: HomeAssistant,
) -> str:
    """
    Render a human-friendly response for a service call.

    Args:
        action: The action from router (e.g., "light.turn_on")
        entity_id: Target entity ID (can be None for scenes/scripts, or "all" for all entities)
        params: Additional parameters from router (brightness, temperature, etc.)
        hass: Home Assistant instance for state lookup

    Returns:
        Human-readable response string

    """
    template_str = RESPONSE_TEMPLATES.get(action, DEFAULT_TEMPLATE)
    template = Template(template_str)

    # Get entity's friendly name from state
    # Handle "all" as a special case for domain-wide operations
    if entity_id and entity_id.lower() == "all":
        # Extract domain from action (e.g., "light.turn_on" -> "light")
        domain = action.split(".")[0] if "." in action else "device"
        friendly_name = f"All {domain}s"
    elif entity_id:
        state = hass.states.get(entity_id)
        if state:
            friendly_name = state.attributes.get("friendly_name", entity_id)
        else:
            friendly_name = entity_id
    else:
        friendly_name = "Device"

    # Build template context
    context = {
        "friendly_name": friendly_name,
        "entity_id": entity_id,
        "action": action,
        **params,
    }

    try:
        return template.render(context)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to render response template: %s", exc)
        return DEFAULT_TEMPLATE


def render_error(
    error_type: str,
    **kwargs: Any,
) -> str:
    """
    Render an error response.

    Args:
        error_type: Key from ERROR_TEMPLATES
        **kwargs: Template context variables

    Returns:
        Human-readable error message

    """
    template_str = ERROR_TEMPLATES.get(error_type, "An error occurred.")
    template = Template(template_str)

    try:
        return template.render(**kwargs)
    except Exception:  # noqa: BLE001
        return "An error occurred."


def get_entity_friendly_name(
    hass: HomeAssistant,
    entity_id: str,
) -> str:
    """
    Get the friendly name of an entity.

    Args:
        hass: Home Assistant instance
        entity_id: The entity ID

    Returns:
        Friendly name or entity_id if not found

    """
    state = hass.states.get(entity_id)
    if state:
        return state.attributes.get("friendly_name", entity_id)
    return entity_id
