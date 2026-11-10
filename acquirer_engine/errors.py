"""One error hierarchy for the engine. The CLI maps each class to a one-line message and an exit code."""

from __future__ import annotations


class EngineError(Exception):
    """Base class: an expected failure with a message fit for the user."""

    exit_code = 1


class DataError(EngineError, ValueError):
    """The CSV cannot be used (missing/empty file, missing core column, no usable rows)."""

    exit_code = 3


class ConfigError(EngineError, ValueError):
    """Bad user input: target profile, feedback name, schema map, settings file, CLI flag combination."""

    exit_code = 4


class LLMError(EngineError, RuntimeError):
    """A model provider failed in a way the run cannot recover from (missing key, unknown provider, ...)."""

    exit_code = 5


class RenderError(EngineError, OSError):
    """Output could not be written (e.g. the output directory is not writable)."""

    exit_code = 6
