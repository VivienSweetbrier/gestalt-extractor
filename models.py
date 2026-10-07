"""
gestalt_extractor.models
========================
Data models and schema definitions for the Gestalt Extractor pipeline.

Provides environment duality:
- If `pydantic` is installed, uses Pydantic BaseModel for rich validation.
- If `pydantic` is absent, seamlessly falls back to Python standard library `@dataclass`.
- In both modes, strictly guarantees the 4-key schema:
  `{"file": str, "domain": str, "bounds": List[ExtractedBound], "summary": str}`
"""

from __future__ import annotations

import os
import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Sequence,
    Union,
    Iterator,
    Type,
    TypeVar,
)

logger = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when extraction data fails validation against the canonical schema."""
    pass

# =============================================================================
# 1. Environment Probing (Pydantic vs Dataclass Duality)
# =============================================================================

PYDANTIC_AVAILABLE: bool = False
PYDANTIC_VERSION: Optional[str] = None
IS_PYDANTIC_V2: bool = False

try:
    import pydantic
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
    PYDANTIC_VERSION = getattr(pydantic, "__version__", "2.0.0")
    IS_PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")
    logger.debug("Gestalt Models: Pydantic %s detected (v2=%s).", PYDANTIC_VERSION, IS_PYDANTIC_V2)
except Exception as e:
    PYDANTIC_AVAILABLE = False
    PYDANTIC_VERSION = None
    IS_PYDANTIC_V2 = False
    logger.debug("Gestalt Models: Pydantic not available (%s). Using standard library dataclasses.", e)


# =============================================================================
# 2. Dataclass Implementation (Stdlib Baseline Tier)
# =============================================================================

@dataclass
class DataclassExtractedBound:
    """Standard library dataclass representation of an extracted mathematical bound."""
    name: str = "Extracted Bound"
    formula: str = ""
    type: str = "algorithmic_bound"
    context: str = ""

    def __init__(
        self,
        name: str = "Extracted Bound",
        formula: str = "",
        type: str = "algorithmic_bound",
        context: str = "",
        **kwargs: Any,
    ) -> None:
        self.name = str(name or "Extracted Bound")
        self.formula = str(formula or "")
        self.type = str(type or "algorithmic_bound")
        self.context = str(context or "")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bound strictly to dictionary format."""
        return {
            "name": self.name,
            "formula": self.formula,
            "type": self.type,
            "context": self.context,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize bound to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Any) -> DataclassExtractedBound:
        """Construct bound from dictionary, string formula, or existing bound object."""
        if isinstance(data, DataclassExtractedBound):
            return data
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        if isinstance(data, str):
            return cls(name="Mathematical Bound", formula=data, type="algorithmic_bound", context="")
        if not isinstance(data, dict):
            return cls(name="Unknown", formula=str(data) if data is not None else "", type="algorithmic_bound", context="")
        return cls(
            name=str(data.get("name") or "Extracted Bound"),
            formula=str(data.get("formula") or ""),
            type=str(data.get("type") or "algorithmic_bound"),
            context=str(data.get("context") or ""),
        )


@dataclass
class DataclassExtractedRecord:
    """Standard library dataclass representation of a manuscript extraction record.

    Guarantees mandatory keys: `file`, `domain`, `bounds`, and `summary`.
    """
    file: str = "unknown_file"
    domain: str = "General_Applied_Math"
    bounds: List[DataclassExtractedBound] = field(default_factory=list)
    summary: str = ""
    path: Optional[str] = None
    methodologies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        file: str = "unknown_file",
        domain: str = "General_Applied_Math",
        bounds: Optional[Sequence[Any]] = None,
        summary: str = "",
        path: Optional[str] = None,
        methodologies: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.file = str(file or "unknown_file")
        self.domain = str(domain or "General_Applied_Math")

        # Normalize bounds list
        norm_bounds: List[DataclassExtractedBound] = []
        if bounds:
            for b in bounds:
                norm_bounds.append(DataclassExtractedBound.from_dict(b))
        self.bounds = norm_bounds

        self.summary = str(summary or "")
        self.path = str(path) if path is not None else None
        self.methodologies = [str(m) for m in methodologies] if methodologies is not None else None
        self.metadata = dict(metadata) if metadata is not None else None

    def to_dict(self, include_optional: bool = True) -> Dict[str, Any]:
        """Convert record strictly ensuring the 4 mandatory keys exist."""
        serialized_bounds = [
            b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else DataclassExtractedBound.from_dict(b).to_dict())
            for b in (self.bounds or [])
        ]
        out: Dict[str, Any] = {
            "file": self.file,
            "domain": self.domain,
            "bounds": serialized_bounds,
            "summary": self.summary,
        }
        if include_optional:
            if self.path is not None:
                out["path"] = self.path
            if self.methodologies is not None:
                out["methodologies"] = self.methodologies
            if self.metadata is not None:
                out["metadata"] = self.metadata
        return out

    def to_json(self, indent: Optional[int] = 2, include_optional: bool = True) -> str:
        """Serialize record to formatted JSON string."""
        return json.dumps(self.to_dict(include_optional=include_optional), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Any) -> DataclassExtractedRecord:
        """Construct record from dictionary with mandatory schema validation."""
        if isinstance(data, DataclassExtractedRecord):
            return data
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        if not isinstance(data, dict):
            raise TypeError(f"Cannot construct ExtractedRecord from {type(data).__name__}")

        required_fields = ("file", "domain", "bounds", "summary")
        for field_name in required_fields:
            if field_name not in data or data[field_name] is None:
                raise SchemaValidationError(
                    f"Schema validation failed: required field '{field_name}' is missing or None in input data."
                )

        return cls(
            file=data["file"],
            domain=data["domain"],
            bounds=data["bounds"],
            summary=data["summary"],
            path=data.get("path"),
            methodologies=data.get("methodologies"),
            metadata=data.get("metadata"),
        )


@dataclass
class DataclassExtractionReport:
    """Standard library dataclass collection of extracted records."""
    records: List[DataclassExtractedRecord] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        records: Optional[Sequence[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        rec_list: List[DataclassExtractedRecord] = []
        if records:
            for r in records:
                rec_list.append(DataclassExtractedRecord.from_dict(r))
        self.records = rec_list
        self.metadata = dict(metadata) if metadata is not None else None

    def add_record(self, record: Union[DataclassExtractedRecord, Dict[str, Any]]) -> None:
        """Append record to the collection."""
        self.records.append(DataclassExtractedRecord.from_dict(record))

    def to_dict(self, as_envelope: bool = False, include_optional: bool = True) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Serialize to standard JSON array (default) or envelope object."""
        rec_dicts = [r.to_dict(include_optional=include_optional) for r in self.records]
        if as_envelope:
            return {
                "metadata": self.metadata or {},
                "records": rec_dicts,
            }
        return rec_dicts

    def to_json(self, indent: Optional[int] = 2, as_envelope: bool = False, include_optional: bool = True) -> str:
        """Serialize report to JSON string."""
        return json.dumps(
            self.to_dict(as_envelope=as_envelope, include_optional=include_optional),
            indent=indent,
            ensure_ascii=False,
        )

    def save(
        self,
        filepath: Union[str, Path],
        indent: Optional[int] = 2,
        as_envelope: bool = False,
        include_optional: bool = True,
    ) -> Path:
        """Save report atomically to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        payload = self.to_json(indent=indent, as_envelope=as_envelope, include_optional=include_optional)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        temp_path.replace(target_path)
        return target_path

    @classmethod
    def from_records(cls, records: Sequence[Any], metadata: Optional[Dict[str, Any]] = None) -> DataclassExtractionReport:
        """Construct report directly from sequence of records."""
        return cls(records=records, metadata=metadata)

    @classmethod
    def from_dict(cls, data: Union[List[Any], Dict[str, Any]]) -> DataclassExtractionReport:
        """Parse report from JSON-deserialized dict or list."""
        if isinstance(data, list):
            return cls(records=data)
        if isinstance(data, dict):
            if "records" in data:
                return cls(records=data.get("records", []), metadata=data.get("metadata"))
            elif "file" in data:
                return cls(records=[data])
            else:
                return cls(records=[], metadata=data)
        raise TypeError(f"Cannot parse ExtractionReport from {type(data).__name__}")

    @classmethod
    def from_json(cls, json_str: str) -> DataclassExtractionReport:
        """Parse report from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> DataclassExtractionReport:
        """Load report from JSON file on disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DataclassExtractedRecord]:
        return iter(self.records)

    def __getitem__(self, index: int) -> DataclassExtractedRecord:
        return self.records[index]

    def __bool__(self) -> bool:
        return bool(self.records)


# =============================================================================
# 3. Pydantic Implementation (Enhanced Validation Tier)
# =============================================================================

if PYDANTIC_AVAILABLE:
    if IS_PYDANTIC_V2:
        from pydantic import ConfigDict, model_validator

        class PydanticExtractedBound(BaseModel):
            """Pydantic v2 representation of an extracted mathematical bound."""
            model_config = ConfigDict(extra="ignore", populate_by_name=True)

            name: str = Field(default="Extracted Bound")
            formula: str = Field(default="")
            type: str = Field(default="algorithmic_bound")
            context: str = Field(default="")

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "name": str(self.name or "Extracted Bound"),
                    "formula": str(self.formula or ""),
                    "type": str(self.type or "algorithmic_bound"),
                    "context": str(self.context or ""),
                }

            def to_json(self, indent: Optional[int] = 2) -> str:
                return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

            @classmethod
            def from_dict(cls, data: Any) -> PydanticExtractedBound:
                if isinstance(data, cls):
                    return data
                if hasattr(data, "to_dict"):
                    data = data.to_dict()
                if isinstance(data, str):
                    return cls(name="Mathematical Bound", formula=data, type="algorithmic_bound", context="")
                if not isinstance(data, dict):
                    return cls(name="Unknown", formula=str(data) if data is not None else "", type="algorithmic_bound", context="")
                return cls(
                    name=str(data.get("name") or "Extracted Bound"),
                    formula=str(data.get("formula") or ""),
                    type=str(data.get("type") or "algorithmic_bound"),
                    context=str(data.get("context") or ""),
                )


        class PydanticExtractedRecord(BaseModel):
            """Pydantic v2 representation of a manuscript extraction record."""
            model_config = ConfigDict(extra="ignore", populate_by_name=True)

            file: str = Field(default="unknown_file")
            domain: str = Field(default="General_Applied_Math")
            bounds: List[PydanticExtractedBound] = Field(default_factory=list)
            summary: str = Field(default="")
            path: Optional[str] = None
            methodologies: Optional[List[str]] = None
            metadata: Optional[Dict[str, Any]] = None

            @model_validator(mode="before")
            @classmethod
            def _normalize_inputs(cls, data: Any) -> Any:
                if not isinstance(data, dict):
                    return data
                raw_bounds = data.get("bounds", [])
                norm_bounds: List[Any] = []
                if isinstance(raw_bounds, (list, tuple)):
                    for b in raw_bounds:
                        if isinstance(b, PydanticExtractedBound):
                            norm_bounds.append(b)
                        elif isinstance(b, str):
                            norm_bounds.append({"name": "Mathematical Bound", "formula": b, "type": "algorithmic_bound", "context": ""})
                        elif isinstance(b, dict):
                            norm_bounds.append(b)
                        elif hasattr(b, "to_dict"):
                            norm_bounds.append(b.to_dict())
                data["bounds"] = norm_bounds
                return data

            def to_dict(self, include_optional: bool = True) -> Dict[str, Any]:
                serialized_bounds = [
                    b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else PydanticExtractedBound.from_dict(b).to_dict())
                    for b in (self.bounds or [])
                ]
                out: Dict[str, Any] = {
                    "file": str(self.file),
                    "domain": str(self.domain),
                    "bounds": serialized_bounds,
                    "summary": str(self.summary),
                }
                if include_optional:
                    if self.path is not None:
                        out["path"] = self.path
                    if self.methodologies is not None:
                        out["methodologies"] = self.methodologies
                    if self.metadata is not None:
                        out["metadata"] = self.metadata
                return out

            def to_json(self, indent: Optional[int] = 2, include_optional: bool = True) -> str:
                return json.dumps(self.to_dict(include_optional=include_optional), indent=indent, ensure_ascii=False)

            @classmethod
            def from_dict(cls, data: Any) -> PydanticExtractedRecord:
                if isinstance(data, cls):
                    return data
                if hasattr(data, "to_dict"):
                    data = data.to_dict()
                if not isinstance(data, dict):
                    raise TypeError(f"Cannot construct ExtractedRecord from {type(data).__name__}")

                required_fields = ("file", "domain", "bounds", "summary")
                for field_name in required_fields:
                    if field_name not in data or data[field_name] is None:
                        raise SchemaValidationError(
                            f"Schema validation failed: required field '{field_name}' is missing or None in input data."
                        )

                return cls.model_validate(data)


        class PydanticExtractionReport(BaseModel):
            """Pydantic v2 collection of extracted records."""
            model_config = ConfigDict(extra="ignore")

            records: List[PydanticExtractedRecord] = Field(default_factory=list)
            metadata: Optional[Dict[str, Any]] = None

            def add_record(self, record: Union[PydanticExtractedRecord, Dict[str, Any]]) -> None:
                if isinstance(record, PydanticExtractedRecord):
                    self.records.append(record)
                else:
                    self.records.append(PydanticExtractedRecord.from_dict(record))

            def to_dict(self, as_envelope: bool = False, include_optional: bool = True) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
                rec_dicts = [r.to_dict(include_optional=include_optional) for r in self.records]
                if as_envelope:
                    return {
                        "metadata": self.metadata or {},
                        "records": rec_dicts,
                    }
                return rec_dicts

            def to_json(self, indent: Optional[int] = 2, as_envelope: bool = False, include_optional: bool = True) -> str:
                return json.dumps(
                    self.to_dict(as_envelope=as_envelope, include_optional=include_optional),
                    indent=indent,
                    ensure_ascii=False,
                )

            def save(
                self,
                filepath: Union[str, Path],
                indent: Optional[int] = 2,
                as_envelope: bool = False,
                include_optional: bool = True,
            ) -> Path:
                target_path = Path(filepath)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
                payload = self.to_json(indent=indent, as_envelope=as_envelope, include_optional=include_optional)
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(payload)
                temp_path.replace(target_path)
                return target_path

            @classmethod
            def from_records(cls, records: Sequence[Any], metadata: Optional[Dict[str, Any]] = None) -> PydanticExtractionReport:
                parsed_records = [PydanticExtractedRecord.from_dict(r) for r in records]
                return cls(records=parsed_records, metadata=metadata)

            @classmethod
            def from_dict(cls, data: Union[List[Any], Dict[str, Any]]) -> PydanticExtractionReport:
                if isinstance(data, list):
                    return cls.from_records(data)
                if isinstance(data, dict):
                    if "records" in data:
                        return cls.from_records(data.get("records", []), metadata=data.get("metadata"))
                    elif "file" in data:
                        return cls.from_records([data])
                    else:
                        return cls(records=[], metadata=data)
                raise TypeError(f"Cannot parse ExtractionReport from {type(data).__name__}")

            @classmethod
            def from_json(cls, json_str: str) -> PydanticExtractionReport:
                return cls.from_dict(json.loads(json_str))

            @classmethod
            def from_file(cls, filepath: Union[str, Path]) -> PydanticExtractionReport:
                with open(filepath, "r", encoding="utf-8") as f:
                    return cls.from_json(f.read())

            def __len__(self) -> int:
                return len(self.records)

            def __iter__(self) -> Iterator[PydanticExtractedRecord]:
                return iter(self.records)

            def __getitem__(self, index: int) -> PydanticExtractedRecord:
                return self.records[index]

            def __bool__(self) -> bool:
                return bool(self.records)

    else:
        # Pydantic v1 implementation
        from pydantic import root_validator

        class PydanticExtractedBound(BaseModel):  # type: ignore[no-redef]
            class Config:
                extra = "ignore"
                allow_population_by_field_name = True

            name: str = "Extracted Bound"
            formula: str = ""
            type: str = "algorithmic_bound"
            context: str = ""

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "name": str(self.name or "Extracted Bound"),
                    "formula": str(self.formula or ""),
                    "type": str(self.type or "algorithmic_bound"),
                    "context": str(self.context or ""),
                }

            def to_json(self, indent: Optional[int] = 2) -> str:
                return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

            @classmethod
            def from_dict(cls, data: Any) -> PydanticExtractedBound:
                if isinstance(data, cls):
                    return data
                if hasattr(data, "to_dict"):
                    data = data.to_dict()
                if isinstance(data, str):
                    return cls(name="Mathematical Bound", formula=data, type="algorithmic_bound", context="")
                if not isinstance(data, dict):
                    return cls(name="Unknown", formula=str(data) if data is not None else "", type="algorithmic_bound", context="")
                return cls(
                    name=str(data.get("name") or "Extracted Bound"),
                    formula=str(data.get("formula") or ""),
                    type=str(data.get("type") or "algorithmic_bound"),
                    context=str(data.get("context") or ""),
                )


        class PydanticExtractedRecord(BaseModel):  # type: ignore[no-redef]
            class Config:
                extra = "ignore"

            file: str = "unknown_file"
            domain: str = "General_Applied_Math"
            bounds: List[PydanticExtractedBound] = Field(default_factory=list)
            summary: str = ""
            path: Optional[str] = None
            methodologies: Optional[List[str]] = None
            metadata: Optional[Dict[str, Any]] = None

            @root_validator(pre=True)
            def _normalize_v1(cls, values: Dict[str, Any]) -> Dict[str, Any]:
                raw_bounds = values.get("bounds", [])
                norm: List[Any] = []
                if isinstance(raw_bounds, (list, tuple)):
                    for b in raw_bounds:
                        if isinstance(b, str):
                            norm.append({"name": "Mathematical Bound", "formula": b, "type": "algorithmic_bound", "context": ""})
                        elif hasattr(b, "to_dict"):
                            norm.append(b.to_dict())
                        else:
                            norm.append(b)
                values["bounds"] = norm
                return values

            def to_dict(self, include_optional: bool = True) -> Dict[str, Any]:
                serialized_bounds = [
                    b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else PydanticExtractedBound.from_dict(b).to_dict())
                    for b in (self.bounds or [])
                ]
                out: Dict[str, Any] = {
                    "file": str(self.file),
                    "domain": str(self.domain),
                    "bounds": serialized_bounds,
                    "summary": str(self.summary),
                }
                if include_optional:
                    if self.path is not None:
                        out["path"] = self.path
                    if self.methodologies is not None:
                        out["methodologies"] = self.methodologies
                    if self.metadata is not None:
                        out["metadata"] = self.metadata
                return out

            def to_json(self, indent: Optional[int] = 2, include_optional: bool = True) -> str:
                return json.dumps(self.to_dict(include_optional=include_optional), indent=indent, ensure_ascii=False)

            @classmethod
            def from_dict(cls, data: Any) -> PydanticExtractedRecord:
                if isinstance(data, cls):
                    return data
                if hasattr(data, "to_dict"):
                    data = data.to_dict()
                if not isinstance(data, dict):
                    raise TypeError(f"Cannot construct ExtractedRecord from {type(data).__name__}")

                required_fields = ("file", "domain", "bounds", "summary")
                for field_name in required_fields:
                    if field_name not in data or data[field_name] is None:
                        raise SchemaValidationError(
                            f"Schema validation failed: required field '{field_name}' is missing or None in input data."
                        )

                return cls.parse_obj(data)


        class PydanticExtractionReport(BaseModel):  # type: ignore[no-redef]
            class Config:
                extra = "ignore"

            records: List[PydanticExtractedRecord] = Field(default_factory=list)
            metadata: Optional[Dict[str, Any]] = None

            def add_record(self, record: Union[PydanticExtractedRecord, Dict[str, Any]]) -> None:
                if isinstance(record, PydanticExtractedRecord):
                    self.records.append(record)
                else:
                    self.records.append(PydanticExtractedRecord.from_dict(record))

            def to_dict(self, as_envelope: bool = False, include_optional: bool = True) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
                rec_dicts = [r.to_dict(include_optional=include_optional) for r in self.records]
                if as_envelope:
                    return {
                        "metadata": self.metadata or {},
                        "records": rec_dicts,
                    }
                return rec_dicts

            def to_json(self, indent: Optional[int] = 2, as_envelope: bool = False, include_optional: bool = True) -> str:
                return json.dumps(
                    self.to_dict(as_envelope=as_envelope, include_optional=include_optional),
                    indent=indent,
                    ensure_ascii=False,
                )

            def save(
                self,
                filepath: Union[str, Path],
                indent: Optional[int] = 2,
                as_envelope: bool = False,
                include_optional: bool = True,
            ) -> Path:
                target_path = Path(filepath)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
                payload = self.to_json(indent=indent, as_envelope=as_envelope, include_optional=include_optional)
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(payload)
                temp_path.replace(target_path)
                return target_path

            @classmethod
            def from_records(cls, records: Sequence[Any], metadata: Optional[Dict[str, Any]] = None) -> PydanticExtractionReport:
                parsed = [PydanticExtractedRecord.from_dict(r) for r in records]
                return cls(records=parsed, metadata=metadata)

            @classmethod
            def from_dict(cls, data: Union[List[Any], Dict[str, Any]]) -> PydanticExtractionReport:
                if isinstance(data, list):
                    return cls.from_records(data)
                if isinstance(data, dict):
                    if "records" in data:
                        return cls.from_records(data.get("records", []), metadata=data.get("metadata"))
                    elif "file" in data:
                        return cls.from_records([data])
                    else:
                        return cls(records=[], metadata=data)
                raise TypeError(f"Cannot parse ExtractionReport from {type(data).__name__}")

            @classmethod
            def from_json(cls, json_str: str) -> PydanticExtractionReport:
                return cls.from_dict(json.loads(json_str))

            @classmethod
            def from_file(cls, filepath: Union[str, Path]) -> PydanticExtractionReport:
                with open(filepath, "r", encoding="utf-8") as f:
                    return cls.from_json(f.read())

            def __len__(self) -> int:
                return len(self.records)

            def __iter__(self) -> Iterator[PydanticExtractedRecord]:
                return iter(self.records)

            def __getitem__(self, index: int) -> PydanticExtractedRecord:
                return self.records[index]

            def __bool__(self) -> bool:
                return bool(self.records)


# =============================================================================
# 4. Canonical Export Resolution & Type Aliases
# =============================================================================

if PYDANTIC_AVAILABLE:
    ExtractedBound = PydanticExtractedBound
    ExtractedRecord = PydanticExtractedRecord
    ExtractionReport = PydanticExtractionReport
else:
    ExtractedBound = DataclassExtractedBound
    ExtractedRecord = DataclassExtractedRecord
    ExtractionReport = DataclassExtractionReport

# Canonical Architectural Aliases across codebase
ExtractionResult = ExtractedRecord
Bound = ExtractedBound
Report = ExtractionReport

__all__ = [
    "PYDANTIC_AVAILABLE",
    "PYDANTIC_VERSION",
    "IS_PYDANTIC_V2",
    "SchemaValidationError",
    "ExtractedBound",
    "ExtractedRecord",
    "ExtractionReport",
    "ExtractionResult",
    "Bound",
    "Report",
    "DataclassExtractedBound",
    "DataclassExtractedRecord",
    "DataclassExtractionReport",
]

if PYDANTIC_AVAILABLE:
    __all__.extend([
        "PydanticExtractedBound",
        "PydanticExtractedRecord",
        "PydanticExtractionReport",
    ])
