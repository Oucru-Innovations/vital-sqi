"""Raw-waveform upload handling for the Dash app's Compute view.

Wraps :func:`vital_sqi.data.signal_io.ECG_reader` /
:func:`vital_sqi.data.signal_io.PPG_reader` so that an in-memory upload
from ``dcc.Upload`` can be turned into a numpy array + sampling rate
without requiring the user to know about file paths.

Returned payload
----------------
Every loader returns a :class:`LoadedWaveform` (a dataclass) with the
fields the Compute callback needs to drive the rest of the pipeline:

    - ``signal``       (np.ndarray, 1D)        - sample values
    - ``timestamps``   (pandas-friendly list)  - optional, may be ``None``
    - ``sampling_rate`` (float, Hz)
    - ``wave_type``    ("ECG" or "PPG")
    - ``source_name``  (original filename, for display)

Errors raise :class:`WaveformLoaderError` so the callback can put a
user-visible message in the UI; the original cause is chained.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class WaveformLoaderError(RuntimeError):
    """Raised when an uploaded recording cannot be parsed.

    The message is intentionally short and end-user-friendly; details that
    only matter to developers (full tracebacks, raw exception types) are
    logged via :data:`logger`.
    """


@dataclass
class LoadedWaveform:
    """Container for a successfully loaded recording."""

    signal: np.ndarray
    sampling_rate: float
    wave_type: str
    source_name: str
    timestamps: Optional[List[str]] = field(default=None)

    def to_store(self) -> dict:
        """Convert to a JSON-serialisable dict suitable for ``dcc.Store``."""
        return {
            "signal": self.signal.tolist(),
            "sampling_rate": float(self.sampling_rate),
            "wave_type": self.wave_type,
            "source_name": self.source_name,
            "timestamps": self.timestamps,
        }

    @classmethod
    def from_store(cls, payload: dict) -> "LoadedWaveform":
        """Rehydrate a stored payload back into a :class:`LoadedWaveform`."""
        return cls(
            signal=np.asarray(payload["signal"], dtype=float),
            sampling_rate=float(payload["sampling_rate"]),
            wave_type=payload["wave_type"],
            source_name=payload["source_name"],
            timestamps=payload.get("timestamps"),
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_CSV_EXT = {".csv", ".tsv", ".txt"}
_EDF_EXT = {".edf", ".bdf"}
_WFDB_EXT = {".hea", ".dat", ".mat"}


def detect_file_type(filename: str) -> str:
    """Return one of ``'csv'``, ``'edf'``, or ``'mit'`` based on the filename.

    The library's underlying readers refer to MIT-WFDB recordings as ``mit``;
    we follow that convention here.
    """
    if not filename:
        raise WaveformLoaderError("No filename was supplied.")
    ext = os.path.splitext(filename.lower())[1]
    if ext in _CSV_EXT:
        return "csv"
    if ext in _EDF_EXT:
        return "edf"
    if ext in _WFDB_EXT:
        return "mit"
    raise WaveformLoaderError(
        f"Unsupported file extension {ext!r}. "
        "Expected CSV (.csv/.tsv/.txt), EDF (.edf/.bdf), or WFDB (.hea/.dat)."
    )


# ---------------------------------------------------------------------------
# Column introspection (for the UI's signal-column picker)
# ---------------------------------------------------------------------------


@dataclass
class ColumnCandidate:
    """A column the user could plausibly select as the signal source."""

    name: str
    kind: str          # "numeric" (flat CSV) or "array" (Oucru row-per-sec)
    sample_size: int   # for "array": elements per row; for "numeric": row count
    preferred: bool    # True if this matches the wave_type's default
    preview: str       # human-readable snippet for the dropdown label


def introspect_columns(
    contents: str, filename: str, wave_type: str
) -> List[ColumnCandidate]:
    """List the columns that could plausibly hold a signal.

    Returns an empty list for binary formats (EDF / WFDB) — the user can't
    pick a column there because the reader produces a single channel by
    design.  CSV uploads return both flat numeric columns and Oucru-style
    array columns, sorted with the wave-type's default first.

    Parameters
    ----------
    contents
        ``data:...;base64,<b64>`` upload payload from ``dcc.Upload``.
    filename
        Original filename, used for type detection.
    wave_type
        ``"ECG"`` or ``"PPG"``; controls which column is marked
        ``preferred=True``.
    """
    try:
        file_type = detect_file_type(filename)
    except WaveformLoaderError:
        return []
    if file_type != "csv":
        return []

    try:
        raw = _decode_data_url(contents)
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return []
    if df.empty:
        return []

    wt = _validate_wave_type(wave_type) if wave_type else "PPG"
    preferred_names = _OUCRU_ECG_COLUMNS if wt == "ECG" else _OUCRU_PPG_COLUMNS
    # Rank: 0 for the first preferred name, 1 for the second, etc.; a large
    # constant for non-preferred columns so they sort after the preferred set.
    rank_by_lower = {name.lower(): i for i, name in enumerate(preferred_names)}

    candidates: List[ColumnCandidate] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        first = series.iloc[0]
        if isinstance(first, str) and (
            first.lstrip().startswith("[")
            or ("," in first and len(_parse_array_cell(first)) >= 2)
        ):
            arr = _parse_array_cell(first)
            if arr.size < 2:
                continue
            candidates.append(
                ColumnCandidate(
                    name=column,
                    kind="array",
                    sample_size=int(arr.size),
                    preferred=column.lower() in rank_by_lower,
                    preview=f"array of {arr.size} samples/row",
                )
            )
        elif pd.api.types.is_numeric_dtype(df[column]):
            candidates.append(
                ColumnCandidate(
                    name=column,
                    kind="numeric",
                    sample_size=int(len(df)),
                    preferred=False,
                    preview=f"numeric, {len(df)} rows",
                )
            )

    # Sort key: preferred set first (ordered by their position in the
    # canonical tuple), then everything else alphabetically.
    def _sort_key(c: ColumnCandidate) -> Tuple[int, int, str]:
        rank = rank_by_lower.get(c.name.lower())
        if rank is not None:
            return (0, rank, c.name.lower())
        return (1, 0, c.name.lower())

    candidates.sort(key=_sort_key)
    return candidates


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_from_upload(
    contents: str,
    filename: str,
    wave_type: str,
    sampling_rate: Optional[float] = None,
    signal_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    sidecar_contents: Optional[List[Tuple[str, str]]] = None,
) -> LoadedWaveform:
    """Parse the upload payload coming from a Dash ``dcc.Upload`` component.

    Parameters
    ----------
    contents
        The data URL produced by ``dcc.Upload`` (``"data:...;base64,<b64>"``).
    filename
        Original filename (used for type detection and the ``source_name``
        field of the returned record).
    wave_type
        ``"ECG"`` or ``"PPG"``.  This is a user choice rather than something
        we can reliably detect from the file alone.
    sampling_rate
        Required for CSV uploads that don't carry a timestamp column.
        Ignored for EDF/WFDB recordings, which expose ``fs`` directly.
    signal_column, timestamp_column
        Optional CSV column names.  If omitted the loader picks the first
        numeric column as the signal and looks for ``timestamps`` /
        ``timestamp`` / ``time`` as the timestamp column.
    sidecar_contents
        For WFDB uploads that span multiple files (``.hea``, ``.dat``,
        ``.mat``), pass the additional ``(filename, data_url)`` tuples here.

    Returns
    -------
    LoadedWaveform
    """
    wave_type = _validate_wave_type(wave_type)
    file_type = detect_file_type(filename)

    if file_type == "csv":
        return _load_csv(
            contents=contents,
            filename=filename,
            wave_type=wave_type,
            sampling_rate=sampling_rate,
            signal_column=signal_column,
            timestamp_column=timestamp_column,
        )

    # EDF / WFDB go through signal_io's path-based readers — we need to
    # spill the bytes to a temp file first.
    with _temp_dir_with_files(
        primary=(filename, contents),
        sidecars=sidecar_contents or [],
    ) as (tmp_dir, primary_path):
        return _load_via_signal_io(
            file_path=primary_path,
            file_type=file_type,
            wave_type=wave_type,
            source_name=filename,
        )


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------


def _load_csv(
    contents: str,
    filename: str,
    wave_type: str,
    sampling_rate: Optional[float],
    signal_column: Optional[str],
    timestamp_column: Optional[str],
) -> LoadedWaveform:
    raw = _decode_data_url(contents)
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        logger.exception("CSV read failure for %s: %s", filename, exc)
        raise WaveformLoaderError(
            "Could not parse CSV.  Make sure the file is a comma-separated "
            "table with a header row."
        ) from exc

    if df.empty:
        raise WaveformLoaderError("CSV is empty.")

    # The Oucru SmartCare / ECG exporter stores one row per second; the signal
    # column is a JSON-array string containing the samples within that second.
    # Detect it before falling through to the flat per-sample CSV path.
    if _looks_like_oucru_csv(df, wave_type, signal_column):
        return _load_oucru_csv(
            df=df,
            filename=filename,
            wave_type=wave_type,
            signal_column=signal_column,
            timestamp_column=timestamp_column,
        )

    ts_col = _resolve_timestamp_column(df, timestamp_column)
    sig_col = _resolve_signal_column(df, signal_column, exclude=ts_col)
    signal = df[sig_col].to_numpy(dtype=float)

    fs, timestamps = _resolve_sampling_rate(df, ts_col, sampling_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise WaveformLoaderError(
            "Sampling rate must be a positive number; received "
            f"{sampling_rate!r}."
        )

    return LoadedWaveform(
        signal=signal,
        sampling_rate=fs,
        wave_type=wave_type,
        source_name=filename,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# Oucru CSV (row-per-second with array-valued signal columns)
# ---------------------------------------------------------------------------

#: Column-name preferences when the wave type is known.
#: First match wins.  Case-insensitive.
_OUCRU_PPG_COLUMNS = ("pleth", "ppg", "red", "ir")
_OUCRU_ECG_COLUMNS = ("ecg", "ecg_signal", "ecg_data")


def _looks_like_oucru_csv(
    df: pd.DataFrame,
    wave_type: str,
    requested_column: Optional[str],
) -> bool:
    """Return True if the DataFrame is an Oucru row-per-second export.

    Heuristic: the chosen signal column's first non-null cell is a string
    that decodes into an array of numbers.  Accepts both the canonical
    bracketed form ``"[1, 2, 3]"`` and the bracket-less comma-separated
    ``"1,2,3"`` style seen in the ECG file's ``acc_x``/``y``/``z`` columns.
    A user-supplied ``signal_column`` is honoured if present.
    """
    candidate = _pick_oucru_signal_column(df, wave_type, requested_column)
    if candidate is None:
        return False
    first = df[candidate].dropna()
    if first.empty:
        return False
    value = first.iloc[0]
    if not isinstance(value, str):
        return False
    text = value.strip()
    if text.startswith("["):
        return True
    # Bracket-less form: must contain commas AND at least one numeric token
    # that's longer than a single character (avoids false positives on
    # multi-word string columns).
    if "," in text and len(_parse_array_cell(text)) >= 2:
        return True
    return False


def _pick_oucru_signal_column(
    df: pd.DataFrame,
    wave_type: str,
    requested: Optional[str],
) -> Optional[str]:
    """Pick the signal column for an Oucru CSV based on wave type / user override."""
    if requested and requested in df.columns:
        return requested
    preferred = _OUCRU_ECG_COLUMNS if wave_type == "ECG" else _OUCRU_PPG_COLUMNS
    lookup = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in lookup:
            return lookup[name]
    return None


def _parse_array_cell(cell: object) -> np.ndarray:
    """Parse a single JSON-list cell into a 1-D float array.

    Accepts the canonical ``[1, 2, 3]`` form plus the bracket-less
    comma-separated ``acc_x``-style values seen in the ECG dataset
    (``"-439,-446,-446"``).  Returns ``None`` for missing values.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.array([], dtype=float)
    text = str(cell).strip()
    if not text:
        return np.array([], dtype=float)
    try:
        if text.startswith("["):
            parsed = json.loads(text)
        else:
            parsed = [float(piece) for piece in text.split(",") if piece.strip()]
        return np.asarray(parsed, dtype=float)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse array cell %r: %s", text[:40], exc)
        return np.array([], dtype=float)


def _load_oucru_csv(
    df: pd.DataFrame,
    filename: str,
    wave_type: str,
    signal_column: Optional[str],
    timestamp_column: Optional[str],
) -> LoadedWaveform:
    """Decode an Oucru row-per-second CSV into a flat (signal, fs) record.

    Each row's array column is concatenated in row order.  The sampling rate
    is inferred from the first row's array length (rows are 1 s apart by
    contract).  Short or empty rows are zero-padded to that reference length
    so the resulting array remains uniformly sampled — the alternative
    (drop short rows) would silently shift timestamps.
    """
    sig_col = _pick_oucru_signal_column(df, wave_type, signal_column)
    if sig_col is None:
        raise WaveformLoaderError(
            f"No {wave_type} signal column found in this Oucru-format CSV. "
            f"Tried: {_OUCRU_ECG_COLUMNS if wave_type == 'ECG' else _OUCRU_PPG_COLUMNS}. "
            f"Available columns: {list(df.columns)}."
        )

    arrays = [_parse_array_cell(cell) for cell in df[sig_col].tolist()]
    # Each row represents one second of data, so the array length on the
    # *first* non-empty row is the sampling rate.  This matches how the
    # Oucru exporters write the file and avoids scanning every row just to
    # pick a number.  Empty / unparseable leading rows are skipped.
    samples_per_second = next((a.size for a in arrays if a.size > 0), 0)
    if samples_per_second <= 0:
        raise WaveformLoaderError(
            f"Column {sig_col!r} contained no parseable arrays."
        )

    # Normalise every row to the modal length: pad short rows with NaN (so
    # downstream code can mask them) and trim over-long ones.  Pure-zero
    # padding would bias SQIs; NaN is the honest signal-missing marker.
    normalised = np.empty((len(arrays), samples_per_second), dtype=float)
    normalised[:] = np.nan
    for i, arr in enumerate(arrays):
        n = min(arr.size, samples_per_second)
        if n:
            normalised[i, :n] = arr[:n]

    flat = normalised.ravel()
    # Replace remaining NaNs with linearly interpolated values so the
    # downstream pipeline doesn't get NaN-poisoned.  If everything is NaN
    # we just keep zeros.
    mask = ~np.isnan(flat)
    if mask.any() and not mask.all():
        idx = np.arange(flat.size)
        flat[~mask] = np.interp(idx[~mask], idx[mask], flat[mask])
    elif not mask.any():
        flat = np.zeros_like(flat)

    # Build per-row timestamps if the file has them; we only emit the row
    # timestamps (one per second), not per-sample, to keep the payload small.
    ts_col = timestamp_column or "timestamp"
    timestamps: Optional[List[str]] = None
    if ts_col in df.columns:
        try:
            parsed_ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            if parsed_ts.notna().any():
                timestamps = parsed_ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z").tolist()
        except Exception as exc:
            logger.warning("Failed to parse Oucru timestamps: %s", exc)

    return LoadedWaveform(
        signal=flat,
        sampling_rate=float(samples_per_second),
        wave_type=wave_type,
        source_name=filename,
        timestamps=timestamps,
    )


def _load_via_signal_io(
    file_path: str,
    file_type: str,
    wave_type: str,
    source_name: str,
) -> LoadedWaveform:
    """EDF / WFDB → delegated to signal_io's typed readers."""
    # Import lazily so vitalDSP/pyedflib aren't required just to import
    # the loader module.
    from vital_sqi.data.signal_io import ECG_reader, PPG_reader

    reader = ECG_reader if wave_type == "ECG" else PPG_reader
    try:
        signal_sqi = reader(file_path, file_type=file_type)
    except Exception as exc:
        logger.exception("%s read failure for %s: %s", file_type, source_name, exc)
        raise WaveformLoaderError(
            f"Could not read the {file_type.upper()} file. "
            "Verify the format and required side-car files."
        ) from exc

    fs = float(signal_sqi.sampling_rate)
    # SignalSQI.signals is a DataFrame with timestamp + one or more channels.
    df = signal_sqi.signals
    if df is None or df.empty:
        raise WaveformLoaderError("Recording contains no samples.")
    sig_col = _resolve_signal_column(df, requested=None, exclude=df.columns[0])
    return LoadedWaveform(
        signal=df[sig_col].to_numpy(dtype=float),
        sampling_rate=fs,
        wave_type=wave_type,
        source_name=source_name,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VALID_WAVE_TYPES = {"ECG", "PPG"}


def _validate_wave_type(wave_type: Optional[str]) -> str:
    if wave_type is None:
        raise WaveformLoaderError("wave_type is required (must be 'ECG' or 'PPG').")
    normalised = wave_type.upper()
    if normalised not in _VALID_WAVE_TYPES:
        raise WaveformLoaderError(
            f"wave_type must be 'ECG' or 'PPG'; received {wave_type!r}."
        )
    return normalised


_TIMESTAMP_CANDIDATES = (
    "timestamps", "timestamp", "time", "datetime", "ts",
    "timestamp_ms", "timestamp_us", "timestamp_ns",
)


def _resolve_timestamp_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested:
        if requested not in df.columns:
            raise WaveformLoaderError(
                f"Timestamp column {requested!r} not found. "
                f"Available columns: {list(df.columns)}."
            )
        return requested
    # Case-insensitive match: build lower-case lookup.
    lookup = {c.lower(): c for c in df.columns}
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _resolve_signal_column(
    df: pd.DataFrame, requested: Optional[str], exclude: Optional[str]
) -> str:
    if requested:
        if requested not in df.columns:
            raise WaveformLoaderError(
                f"Signal column {requested!r} not found. "
                f"Available columns: {list(df.columns)}."
            )
        return requested
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude in numeric:
        numeric.remove(exclude)
    if not numeric:
        raise WaveformLoaderError(
            "No numeric signal column found in the file. "
            f"Columns: {list(df.columns)}."
        )
    return numeric[0]


def _resolve_sampling_rate(
    df: pd.DataFrame,
    ts_col: Optional[str],
    explicit: Optional[float],
) -> Tuple[float, Optional[List[str]]]:
    """Pick a sampling rate using (1) explicit input, (2) timestamp column, (3) error."""
    timestamps: Optional[List[str]] = None
    if ts_col is not None:
        # Numeric timestamp columns may be in ms / us / ns since epoch; the
        # column name is the only hint (TIMESTAMP_MS / _US / _NS / plain
        # "timestamps" treated as seconds).
        unit_scale = _numeric_timestamp_unit_scale(ts_col)
        # Numeric columns (any int / float dtype) go through the numeric
        # branch with `unit_scale`.  ``pd.to_datetime`` would otherwise
        # interpret a column of seconds-as-floats as nanoseconds since epoch,
        # collapsing every value to 1970-01-01 with zero deltas.
        if pd.api.types.is_numeric_dtype(df[ts_col]):
            ts_series = pd.to_numeric(df[ts_col], errors="coerce")
        else:
            try:
                ts_series = pd.to_datetime(df[ts_col], errors="raise", utc=False)
            except Exception:
                ts_series = pd.to_numeric(df[ts_col], errors="coerce")
        if isinstance(ts_series.dtype, pd.DatetimeTZDtype) or np.issubdtype(
            ts_series.dtype, np.datetime64
        ):
            deltas = ts_series.diff().dropna().dt.total_seconds().to_numpy()
            timestamps = ts_series.astype(str).tolist()
        else:
            deltas = np.diff(ts_series.to_numpy(dtype=float)) * unit_scale
        deltas = deltas[deltas > 0]
        if len(deltas) > 0:
            median_step = float(np.median(deltas))
            if median_step > 0:
                return 1.0 / median_step, timestamps
    if explicit is None:
        raise WaveformLoaderError(
            "CSV does not contain a timestamp column and no sampling rate was "
            "supplied.  Add a 'timestamps' column or set the sampling rate in "
            "the form."
        )
    return float(explicit), timestamps


def _numeric_timestamp_unit_scale(column: str) -> float:
    """Multiplier that turns the column's numeric values into seconds.

    Defaults to 1.0 (seconds) unless the column name carries a recognised
    suffix.
    """
    lower = column.lower()
    if lower.endswith("_ns"):
        return 1e-9
    if lower.endswith("_us"):
        return 1e-6
    if lower.endswith("_ms"):
        return 1e-3
    return 1.0


def _decode_data_url(data_url: str) -> bytes:
    """Strip the ``data:...;base64,`` prefix and decode the body."""
    try:
        _, b64 = data_url.split(",", 1)
    except ValueError as exc:
        raise WaveformLoaderError("Upload payload is malformed.") from exc
    try:
        return base64.b64decode(b64)
    except Exception as exc:
        raise WaveformLoaderError("Upload payload is not valid base64.") from exc


@contextmanager
def _temp_dir_with_files(
    primary: Tuple[str, str],
    sidecars: List[Tuple[str, str]],
):
    """Write the primary upload (and any sidecar files) to a temp directory.

    Returns the directory path and the absolute path of the primary file.
    For WFDB the file_path passed to ``rdsamp`` is the *base* (no extension),
    so we strip the extension for callers that need it.
    """
    tmp_dir = tempfile.mkdtemp(prefix="vital_sqi_upload_")
    try:
        primary_name, primary_url = primary
        primary_path = os.path.join(tmp_dir, primary_name)
        with open(primary_path, "wb") as fh:
            fh.write(_decode_data_url(primary_url))
        for side_name, side_url in sidecars:
            side_path = os.path.join(tmp_dir, side_name)
            with open(side_path, "wb") as fh:
                fh.write(_decode_data_url(side_url))
        # For WFDB the library wants the record stem (no extension).
        stem, ext = os.path.splitext(primary_path)
        wfdb_path = stem if ext.lower() in (".hea", ".dat") else primary_path
        yield tmp_dir, wfdb_path
    finally:
        # Best-effort cleanup; ignore errors so a Windows lock doesn't crash
        # the callback.
        for name in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
