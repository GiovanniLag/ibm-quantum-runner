"""Structured validation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    circuit_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "circuit_index": self.circuit_index,
            "details": self.details,
        }


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    backend_name: str | None = None
    transpiled_metadata: list[dict[str, Any]] = field(default_factory=list)
    backend_selection: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        circuit_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(severity, code, message, circuit_index, details or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "is_valid": self.is_valid,
            "backend_name": self.backend_name,
            "backend_selection": self.backend_selection,
            "transpiled_metadata": self.transpiled_metadata,
            "issues": [issue.to_dict() for issue in self.issues],
        }
