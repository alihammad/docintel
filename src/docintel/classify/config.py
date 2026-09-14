"""YAML configuration for classification (LLM settings + category taxonomy)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from docintel.classify.categories import CATEGORY_KEYWORDS, DEFAULT_CATEGORIES

CONFIG_FILENAMES = ("docintel.yaml", "docintel.yml")


def default_config_paths() -> list[Path]:
    """Config search order: cwd first, then XDG/user config dir."""
    paths = [Path.cwd() / name for name in CONFIG_FILENAMES]
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    paths.append(config_home / "docintel" / "config.yaml")
    paths.append(config_home / "docintel" / "config.yml")
    return paths


def find_config_file(explicit: Path | None = None) -> Path | None:
    """Return the config file to use, or None if no config exists."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    for path in default_config_paths():
        if path.is_file():
            return path
    return None


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str | None = None  # no hardcoded default; must come from config/env
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


@dataclass
class ClassificationConfig:
    top_n: int = 5
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    keywords: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in CATEGORY_KEYWORDS.items()}
    )


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    source: Path | None = None  # where the config was loaded from (None = defaults)


class ConfigError(ValueError):
    """Raised when a config file is present but invalid."""


def load_config(explicit_path: Path | None = None) -> Config:
    """Load config from YAML, falling back to built-in defaults.

    Raises :class:`ConfigError` if an explicit path was given but not found,
    or if the YAML content is malformed.
    """
    path = find_config_file(explicit_path)
    if explicit_path is not None and path is None:
        raise ConfigError(f"config file not found: {explicit_path}")

    config = Config(source=path)
    if path is None:
        return _apply_env_overrides(config)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    llm_raw = raw.get("llm", {})
    if not isinstance(llm_raw, dict):
        raise ConfigError("'llm' section must be a mapping")
    llm = config.llm
    for key in ("provider", "model", "api_key_env", "base_url"):
        if key in llm_raw and llm_raw[key] is not None:
            setattr(llm, key, str(llm_raw[key]))

    cls_raw = raw.get("classification", {})
    if not isinstance(cls_raw, dict):
        raise ConfigError("'classification' section must be a mapping")
    cls = config.classification
    if "top_n" in cls_raw:
        try:
            cls.top_n = max(1, int(cls_raw["top_n"]))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"top_n must be an integer: {cls_raw['top_n']!r}") from exc
    if "categories" in cls_raw:
        cats = cls_raw["categories"]
        if not isinstance(cats, list) or not all(isinstance(c, str) for c in cats):
            raise ConfigError("'categories' must be a list of strings")
        if not cats:
            raise ConfigError("'categories' must not be empty")
        cls.categories = list(cats)
    if "keywords" in cls_raw:
        kw = cls_raw["keywords"]
        if not isinstance(kw, dict):
            raise ConfigError("'keywords' must be a mapping of category -> list of phrases")
        for category, phrases in kw.items():
            if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
                raise ConfigError(f"keywords[{category!r}] must be a list of strings")
            key = str(category)
            # User keywords extend (never replace) the built-in hints.
            existing = cls.keywords.get(key, ())
            added = tuple(dict.fromkeys(p.lower() for p in phrases))
            cls.keywords[key] = tuple(dict.fromkeys([*existing, *added]))

    return _apply_env_overrides(config)


def _apply_env_overrides(config: Config) -> Config:
    """Environment variables take precedence over the config file."""
    if model := os.environ.get("DOCINTEL_MODEL"):
        config.llm.model = model
    if base_url := os.environ.get("DOCINTEL_BASE_URL"):
        config.llm.base_url = base_url
    if api_key_env := os.environ.get("DOCINTEL_API_KEY_ENV"):
        config.llm.api_key_env = api_key_env
    return config
