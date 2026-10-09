"""Logging helpers that never configure the application's root logger."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

LOGGER_NAME = "ibm_quantum_runner"
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER.addHandler(logging.NullHandler())


class ExecutionLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Attach execution and IBM job identifiers to operational records."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = self.extra or {}
        identifiers = " ".join(
            f"{key}={value}" for key in ("execution_id", "ibm_job_id") if (value := extra.get(key))
        )
        return (f"[{identifiers}] {msg}" if identifiers else msg), kwargs


def get_logger(
    *, execution_id: str | None = None, ibm_job_id: str | None = None
) -> ExecutionLoggerAdapter:
    return ExecutionLoggerAdapter(
        LOGGER,
        {"execution_id": execution_id, "ibm_job_id": ibm_job_id},
    )


def execution_logger(
    logger: logging.Logger,
    execution_id: str,
    ibm_job_id: str | None = None,
) -> ExecutionLoggerAdapter:
    """Wrap a module logger with non-secret execution identifiers."""

    return ExecutionLoggerAdapter(
        logger,
        {"execution_id": execution_id, "ibm_job_id": ibm_job_id},
    )


def set_package_log_level(level: str | int) -> None:
    """Set only the package logger level; handlers remain caller-controlled."""

    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"Invalid log level: {level!r}")
        level = resolved
    LOGGER.setLevel(level)
