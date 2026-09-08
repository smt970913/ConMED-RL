"""Pluggable LLM backends for clinical-term translation and variable search.

Databases curated outside the English-speaking world label their signals in the
local language -- SICdb, for instance, mixes English (``HeartRateECG``) with
German (``Herzchirurgie``). Searching such a dictionary for "heart rate" with a
plain substring match silently misses half the relevant signals.

This module wraps that translation step behind one small interface so the rest
of the pipeline never imports a vendor SDK:

>>> backend = get_llm_backend(LLMConfig(provider="openai", api_key="sk-..."))
>>> backend.translate("heart rate", "German")
'Herzfrequenz'

Only *variable names and units* are ever sent to a provider; patient-level data
never leaves the machine.
"""

from __future__ import annotations

import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import LLMConfig

__all__ = [
    "LLMBackend",
    "NullLLMBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "get_llm_backend",
    "MEDICAL_GLOSSARY_EN_DE",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Offline glossary
# ---------------------------------------------------------------------------

#: Hand-curated English -> German ICU terminology. Used by
#: :class:`NullLLMBackend`, and as a cache seed for the online backends so
#: common terms never cost a token.
MEDICAL_GLOSSARY_EN_DE: Dict[str, Tuple[str, ...]] = {
    "heart rate": ("Herzfrequenz", "Herzschlag", "Puls"),
    "heart": ("Herz",),
    "blood pressure": ("Blutdruck",),
    "systolic": ("systolisch", "Systole"),
    "diastolic": ("diastolisch", "Diastole"),
    "mean arterial pressure": ("Mittlerer arterieller Druck", "arterieller Mitteldruck"),
    "respiratory rate": ("Atemfrequenz", "Atemrate"),
    "respiration": ("Atmung", "Beatmung"),
    "temperature": ("Temperatur", "Körpertemperatur"),
    "oxygen saturation": ("Sauerstoffsättigung", "Sättigung"),
    "oxygen": ("Sauerstoff",),
    "tidal volume": ("Tidalvolumen", "Atemzugvolumen"),
    "minute volume": ("Minutenvolumen",),
    "peak pressure": ("Spitzendruck", "Beatmungsdruck"),
    "ventilation": ("Beatmung", "Ventilation"),
    "ventilator": ("Respirator", "Beatmungsgerät"),
    "extubation": ("Extubation",),
    "intubation": ("Intubation",),
    "weight": ("Gewicht", "Körpergewicht"),
    "height": ("Größe", "Körpergröße"),
    "age": ("Alter",),
    "sex": ("Geschlecht",),
    "gender": ("Geschlecht",),
    "hemoglobin": ("Hämoglobin",),
    "hematocrit": ("Hämatokrit",),
    "platelet count": ("Thrombozyten", "Blutplättchen"),
    "white blood cell": ("Leukozyten", "weiße Blutkörperchen"),
    "creatinine": ("Kreatinin",),
    "urea": ("Harnstoff",),
    "sodium": ("Natrium",),
    "potassium": ("Kalium",),
    "chloride": ("Chlorid",),
    "calcium": ("Kalzium", "Calcium"),
    "magnesium": ("Magnesium",),
    "glucose": ("Glukose", "Blutzucker", "Glucose"),
    "lactate": ("Laktat",),
    "bilirubin": ("Bilirubin",),
    "albumin": ("Albumin",),
    "bicarbonate": ("Bikarbonat", "Standardbikarbonat"),
    "base excess": ("Basenüberschuss", "Basenabweichung"),
    "prothrombin time": ("Prothrombinzeit", "Quick"),
    "partial thromboplastin time": ("partielle Thromboplastinzeit", "PTT"),
    "urine output": ("Urinausscheidung", "Harnausscheidung", "Diurese"),
    "length of stay": ("Aufenthaltsdauer", "Liegedauer"),
    "readmission": ("Wiederaufnahme",),
    "mortality": ("Sterblichkeit", "Mortalität"),
    "death": ("Tod", "Exitus"),
    "discharge": ("Entlassung", "Verlegung"),
    "admission": ("Aufnahme",),
    "sedation": ("Sedierung",),
    "consciousness": ("Bewusstsein",),
    "glasgow coma scale": ("Glasgow-Koma-Skala",),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _split_camel(text: str) -> str:
    """``HeartRateECG`` -> ``Heart Rate ECG`` so word-level matching works."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return re.sub(r"[_\-/]+", " ", spaced)


# ---------------------------------------------------------------------------
# Translation cache
# ---------------------------------------------------------------------------


class _TranslationCache:
    """Tiny JSON-backed cache keyed by ``(text, target_language)``."""

    def __init__(self, path: Optional[Path]) -> None:
        self._path = Path(path) if path is not None else None
        self._store: Dict[str, str] = {}
        if self._path is not None and self._path.is_file():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._store.update({str(k): str(v) for k, v in loaded.items()})
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable translation cache %s: %s", self._path, exc)

    @staticmethod
    def _key(text: str, language: str) -> str:
        return "{0}\u241f{1}".format(_normalize(text), _normalize(language))

    def get(self, text: str, language: str) -> Optional[str]:
        return self._store.get(self._key(text, language))

    def put(self, text: str, language: str, value: str) -> None:
        self._store[self._key(text, language)] = value
        self.flush()

    def flush(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._store, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Could not persist translation cache %s: %s", self._path, exc)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class LLMBackend(object):
    """Common interface for term translation and candidate ranking."""

    name = "base"
    #: ``False`` for the offline backend, so callers can report honestly
    #: whether an LLM actually contributed.
    is_llm = False

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self._cache = _TranslationCache(getattr(self.config, "cache_path", None))

    # -- public API -----------------------------------------------------------

    def translate(self, text: str, target_language: str) -> str:
        """Translate one clinical term. Returns ``text`` unchanged on failure."""
        cached = self._cache.get(text, target_language)
        if cached is not None:
            return cached

        glossary_hit = self._glossary_lookup(text, target_language)
        if glossary_hit is not None:
            self._cache.put(text, target_language, glossary_hit)
            return glossary_hit

        try:
            translated = self._translate_impl(text, target_language)
        except Exception as exc:  # noqa: BLE001 - never break a pipeline on this
            logger.warning(
                "%s translation of %r into %s failed: %s", self.name, text, target_language, exc
            )
            return str(text)

        translated = str(translated).strip() or str(text)
        self._cache.put(text, target_language, translated)
        return translated

    def translate_many(
        self, texts: Iterable[str], target_languages: Sequence[str]
    ) -> Dict[str, Dict[str, str]]:
        """Translate several terms into several languages.

        Returns ``{original_term: {language: translation}}``.
        """
        out: Dict[str, Dict[str, str]] = {}
        for text in texts:
            per_language: Dict[str, str] = {}
            for language in target_languages:
                per_language[language] = self.translate(text, language)
            out[str(text)] = per_language
        return out

    def rank_candidates(
        self, query: str, candidates: Sequence[str], top_k: int = 25
    ) -> List[Tuple[str, float]]:
        """Score ``candidates`` by clinical relevance to ``query``.

        Returns ``(candidate, score)`` sorted best-first, scores in ``[0, 1]``.
        """
        if not candidates:
            return []
        try:
            ranked = self._rank_impl(query, candidates, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s ranking for %r failed: %s", self.name, query, exc)
            ranked = None
        if not ranked:
            ranked = _lexical_rank(query, candidates, top_k)
        return ranked

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """Free-form question, used by the interactive search assistant."""
        try:
            return self._chat_impl(prompt, system)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s chat failed: %s", self.name, exc)
            return ""

    def chat_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        validator: Optional[Callable[[Mapping[str, Any]], Any]] = None,
        max_attempts: int = 2,
    ) -> Dict[str, Any]:
        """Request one JSON object and retry only after local validation fails."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        current_prompt = str(prompt)
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            raw = self.chat(current_prompt, system=system).strip()
            if raw.startswith("```") and raw.endswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("response must be a JSON object")
                if validator is not None:
                    validator(payload)
                return payload
            except (TypeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < max_attempts:
                    current_prompt = (
                        str(prompt)
                        + "\nThe previous response failed local validation: "
                        + type(exc).__name__
                        + ". Return a corrected JSON object only."
                    )
        raise ValueError("LLM did not return valid schema-constrained JSON") from last_error

    # -- hooks for subclasses -------------------------------------------------

    def _translate_impl(self, text: str, target_language: str) -> str:
        return str(text)

    def _rank_impl(
        self, query: str, candidates: Sequence[str], top_k: int
    ) -> Optional[List[Tuple[str, float]]]:
        return None

    def _chat_impl(self, prompt: str, system: Optional[str]) -> str:
        return ""

    # -- shared helpers -------------------------------------------------------

    @staticmethod
    def _glossary_lookup(text: str, target_language: str) -> Optional[str]:
        if _normalize(target_language) not in ("german", "de", "deutsch"):
            return None
        key = _normalize(_split_camel(text))
        entry = MEDICAL_GLOSSARY_EN_DE.get(key)
        if entry:
            return entry[0]
        return None


def _lexical_rank(
    query: str, candidates: Sequence[str], top_k: int
) -> List[Tuple[str, float]]:
    """Deterministic fallback ranking: token overlap + fuzzy string similarity.

    Also expands the query through the built-in glossary so an English query
    still scores German candidates.
    """
    query_norm = _normalize(_split_camel(query))
    query_tokens = set(query_norm.split())

    expansions = {query_norm}
    for english, germans in MEDICAL_GLOSSARY_EN_DE.items():
        if english in query_norm or query_norm in english:
            expansions.update(_normalize(g) for g in germans)
            expansions.add(english)

    scored: List[Tuple[str, float]] = []
    for candidate in candidates:
        cand_norm = _normalize(_split_camel(candidate))
        cand_tokens = set(cand_norm.split())

        best = 0.0
        for variant in expansions:
            variant_tokens = set(variant.split())
            if not variant_tokens:
                continue
            overlap = len(query_tokens & cand_tokens) / max(len(variant_tokens), 1)
            substring = 1.0 if variant and variant in cand_norm else 0.0
            fuzzy = SequenceMatcher(None, variant, cand_norm).ratio()
            best = max(best, 0.5 * substring + 0.3 * min(overlap, 1.0) + 0.2 * fuzzy)

        if best > 0.0:
            scored.append((str(candidate), round(min(best, 1.0), 4)))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


class NullLLMBackend(LLMBackend):
    """Offline backend: glossary translation plus fuzzy lexical ranking.

    Chosen automatically when no provider is configured, so the whole
    preprocessing pipeline runs end-to-end without network access or an API key.
    """

    name = "offline"
    is_llm = False

    def _translate_impl(self, text: str, target_language: str) -> str:
        # The glossary was already consulted in `translate`; nothing else to try.
        return str(text)

    def _rank_impl(
        self, query: str, candidates: Sequence[str], top_k: int
    ) -> Optional[List[Tuple[str, float]]]:
        return _lexical_rank(query, candidates, top_k)

    def _chat_impl(self, prompt: str, system: Optional[str]) -> str:
        return (
            "No LLM provider is configured, so the interactive assistant is "
            "unavailable. Pass an LLMConfig with provider='openai' or "
            "'anthropic' to enable it."
        )


_TRANSLATE_SYSTEM = (
    "You translate clinical and physiological variable names used in intensive "
    "care databases. Reply with the translation only: no quotes, no explanation, "
    "no trailing punctuation. Preserve standard clinical abbreviations "
    "(e.g. GCS, PEEP, FiO2, SpO2) unchanged."
)

_RANK_SYSTEM = (
    "You map clinical concepts onto variable names from an intensive care "
    "database dictionary. The names may be English, German, abbreviated or "
    "camel-cased. Reply with JSON only."
)


def _rank_prompt(query: str, candidates: Sequence[str]) -> str:
    listing = "\n".join("- {0}".format(c) for c in candidates)
    return (
        "Clinical concept: {query!r}\n\n"
        "Candidate variable names:\n{listing}\n\n"
        "Return a JSON array of the candidates that plausibly measure this "
        "concept, most relevant first, at most 25 entries. Each element must be "
        'an object {{"name": <candidate exactly as given>, "score": <0.0-1.0>}}. '
        "Return [] if none are relevant. JSON only."
    ).format(query=query, listing=listing)


def _parse_ranking(raw: str, candidates: Sequence[str]) -> Optional[List[Tuple[str, float]]]:
    """Pull a ``[{"name":..., "score":...}]`` array out of a model reply."""
    if not raw:
        return None
    text = raw.strip()
    # Models like to wrap JSON in ``` fences despite instructions.
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(payload, list):
        return None

    # Only accept names the model was actually offered.
    lookup = {_normalize(c): str(c) for c in candidates}
    out: List[Tuple[str, float]] = []
    for item in payload:
        if isinstance(item, dict):
            name, score = item.get("name"), item.get("score", 1.0)
        else:
            name, score = item, 1.0
        resolved = lookup.get(_normalize(name)) if name is not None else None
        if resolved is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 1.0
        out.append((resolved, max(0.0, min(1.0, score_f))))

    out.sort(key=lambda item: (-item[1], item[0]))
    return out or None


class _HTTPBackend(LLMBackend):
    """Shared retry/backoff plumbing for the network-backed providers."""

    def __init__(self, config: LLMConfig) -> None:
        super(_HTTPBackend, self).__init__(config)
        self.api_key = config.resolve_api_key()
        if not self.api_key:
            raise ValueError(
                "No API key for provider {0!r}. Pass LLMConfig(api_key=...) or set "
                "one of the environment variables CONMEDRL_LLM_API_KEY / "
                "{1}.".format(config.provider, config.provider.upper() + "_API_KEY")
            )
        self._client = self._build_client()

    def _build_client(self) -> Any:
        raise NotImplementedError

    def _complete(self, system: Optional[str], prompt: str) -> str:
        raise NotImplementedError

    def _with_retries(self, system: Optional[str], prompt: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, self.config.max_retries)):
            try:
                return self._complete(system, prompt)
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise broadly
                last_exc = exc
                if attempt + 1 >= max(1, self.config.max_retries):
                    break
                delay = 2.0 ** attempt
                logger.debug(
                    "%s call failed (attempt %d/%d): %s; retrying in %.0fs",
                    self.name, attempt + 1, self.config.max_retries, exc, delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            "{0} request failed after {1} attempt(s): {2}".format(
                self.name, max(1, self.config.max_retries), last_exc
            )
        )

    # -- LLMBackend hooks -----------------------------------------------------

    def _translate_impl(self, text: str, target_language: str) -> str:
        prompt = "Translate into {0}: {1}".format(target_language, _split_camel(text))
        return self._with_retries(_TRANSLATE_SYSTEM, prompt)

    def _rank_impl(
        self, query: str, candidates: Sequence[str], top_k: int
    ) -> Optional[List[Tuple[str, float]]]:
        # Keep prompts bounded; beyond a few hundred names the lexical
        # pre-filter is both cheaper and about as good.
        pool = list(candidates)
        if len(pool) > 400:
            pool = [name for name, _ in _lexical_rank(query, pool, 400)] or pool[:400]
        raw = self._with_retries(_RANK_SYSTEM, _rank_prompt(query, pool))
        parsed = _parse_ranking(raw, pool)
        return parsed[:top_k] if parsed else None

    def _chat_impl(self, prompt: str, system: Optional[str]) -> str:
        return self._with_retries(system, prompt)


class OpenAIBackend(_HTTPBackend):
    """OpenAI chat-completions backend (``pip install openai``)."""

    name = "openai"
    is_llm = True

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The OpenAI backend needs the 'openai' package: pip install openai"
            ) from exc
        kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.config.timeout}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return OpenAI(**kwargs)

    def _complete(self, system: Optional[str], prompt: str) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        return (response.choices[0].message.content or "").strip()


class AnthropicBackend(_HTTPBackend):
    """Anthropic Messages backend (``pip install anthropic``)."""

    name = "anthropic"
    is_llm = True

    def _build_client(self) -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The Anthropic backend needs the 'anthropic' package: "
                "pip install anthropic"
            ) from exc
        kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.config.timeout}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return anthropic.Anthropic(**kwargs)

    def _complete(self, system: Optional[str], prompt: str) -> str:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=2048,
            temperature=self.config.temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ]
        return "".join(parts).strip()


_BACKENDS = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
}


def get_llm_backend(config: Optional[LLMConfig] = None) -> LLMBackend:
    """Build the backend described by ``config``.

    Falls back to :class:`NullLLMBackend` when no provider is configured, or
    when the requested provider cannot be constructed (missing SDK, missing
    key) -- preprocessing should degrade rather than crash.
    """
    config = config or LLMConfig()
    if not config.enabled:
        return NullLLMBackend(config)

    backend_cls = _BACKENDS[config.provider]
    try:
        return backend_cls(config)
    except (ImportError, ValueError) as exc:
        logger.warning(
            "Falling back to the offline backend: could not initialise %s (%s)",
            config.provider, exc,
        )
        return NullLLMBackend(config)
