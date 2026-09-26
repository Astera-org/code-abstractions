"""Custom exception hierarchy for the genproc package."""


class GenprocException(Exception):
    """Base exception for Genproc."""


class ConfigValidationError(GenprocException):
    """Exception raised when a config is invalid."""


class DeviceResolutionError(GenprocException):
    """Exception raised when a device resolution fails."""
