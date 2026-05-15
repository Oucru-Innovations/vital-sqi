"""Unit tests for vital_sqi.app.util.waveform_loader."""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import pandas as pd
import pytest

from vital_sqi.app.util.waveform_loader import (
    LoadedWaveform,
    WaveformLoaderError,
    detect_file_type,
    introspect_columns,
    load_from_upload,
)


def _data_url(payload: bytes, mime: str = "text/csv") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def _csv_url(df: pd.DataFrame) -> str:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return _data_url(buf.getvalue())


# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("recording.csv", "csv"),
        ("recording.tsv", "csv"),
        ("recording.txt", "csv"),
        ("recording.edf", "edf"),
        ("recording.bdf", "edf"),
        ("a103l.hea", "mit"),
        ("a103l.dat", "mit"),
        ("a103l.mat", "mit"),
    ],
)
def test_detect_known_extensions(filename, expected):
    assert detect_file_type(filename) == expected


def test_detect_rejects_unknown_extension():
    with pytest.raises(WaveformLoaderError):
        detect_file_type("recording.xyz")


def test_detect_rejects_empty_filename():
    with pytest.raises(WaveformLoaderError):
        detect_file_type("")


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_csv_with_timestamps_seconds(self):
        df = pd.DataFrame(
            {
                "timestamps": np.arange(0, 1, 0.01),
                "signal": np.sin(np.linspace(0, 10, 100)),
            }
        )
        lw = load_from_upload(_csv_url(df), "rec.csv", wave_type="PPG")
        assert lw.signal.shape == (100,)
        assert np.isclose(lw.sampling_rate, 100.0)
        assert lw.wave_type == "PPG"
        assert lw.source_name == "rec.csv"

    def test_csv_with_millisecond_timestamps(self):
        df = pd.DataFrame(
            {
                "TIMESTAMP_MS": np.arange(0, 100, 10),
                "PLETH": np.sin(np.linspace(0, 10, 10)),
            }
        )
        lw = load_from_upload(_csv_url(df), "rec.csv", wave_type="PPG")
        assert lw.signal.shape == (10,)
        assert np.isclose(lw.sampling_rate, 100.0)

    def test_csv_with_microsecond_timestamps(self):
        df = pd.DataFrame(
            {
                "timestamp_us": np.arange(0, 10000, 1000),
                "signal": np.zeros(10),
            }
        )
        lw = load_from_upload(_csv_url(df), "rec.csv", wave_type="ECG")
        assert np.isclose(lw.sampling_rate, 1000.0)

    def test_csv_with_explicit_sampling_rate_overrides_missing_ts(self):
        df = pd.DataFrame({"signal": np.zeros(100)})
        lw = load_from_upload(
            _csv_url(df), "rec.csv", wave_type="ECG", sampling_rate=250
        )
        assert lw.sampling_rate == 250.0

    def test_csv_without_ts_and_no_fs_raises(self):
        df = pd.DataFrame({"signal": np.zeros(100)})
        with pytest.raises(WaveformLoaderError, match="sampling rate"):
            load_from_upload(_csv_url(df), "rec.csv", wave_type="ECG")

    def test_csv_explicit_signal_column(self):
        df = pd.DataFrame(
            {
                "timestamps": np.arange(0, 1, 0.01),
                "lead_I": np.sin(np.linspace(0, 10, 100)),
                "lead_II": np.cos(np.linspace(0, 10, 100)),
            }
        )
        lw = load_from_upload(
            _csv_url(df), "rec.csv", wave_type="ECG", signal_column="lead_II"
        )
        assert np.allclose(lw.signal, np.cos(np.linspace(0, 10, 100)))

    def test_csv_unknown_signal_column_raises(self):
        df = pd.DataFrame({"timestamps": [0, 1], "signal": [0.1, 0.2]})
        with pytest.raises(WaveformLoaderError, match="not found"):
            load_from_upload(
                _csv_url(df),
                "rec.csv",
                wave_type="ECG",
                signal_column="nope",
            )

    def test_empty_csv_raises(self):
        df = pd.DataFrame(columns=["timestamps", "signal"])
        with pytest.raises(WaveformLoaderError):
            load_from_upload(_csv_url(df), "rec.csv", wave_type="PPG")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unsupported_wave_type(self):
        with pytest.raises(WaveformLoaderError, match="wave_type"):
            load_from_upload(
                _csv_url(pd.DataFrame({"x": [1]})),
                "a.csv",
                wave_type="EEG",
                sampling_rate=100,
            )

    def test_unsupported_extension(self):
        with pytest.raises(WaveformLoaderError):
            load_from_upload(
                _data_url(b"junk", mime="application/octet-stream"),
                "rec.xyz",
                wave_type="PPG",
            )

    def test_malformed_data_url(self):
        with pytest.raises(WaveformLoaderError, match="malformed"):
            load_from_upload("not-a-data-url", "rec.csv", wave_type="PPG")

    def test_wave_type_case_insensitive(self):
        df = pd.DataFrame({"timestamps": [0, 0.01], "signal": [0.1, 0.2]})
        lw = load_from_upload(_csv_url(df), "rec.csv", wave_type="ppg")
        assert lw.wave_type == "PPG"


# ---------------------------------------------------------------------------
# LoadedWaveform serialisation round-trip
# ---------------------------------------------------------------------------


class TestStoreRoundTrip:
    def test_round_trip(self):
        lw = LoadedWaveform(
            signal=np.array([1.0, 2.0, 3.0]),
            sampling_rate=250.0,
            wave_type="ECG",
            source_name="x.csv",
            timestamps=None,
        )
        payload = lw.to_store()
        restored = LoadedWaveform.from_store(payload)
        np.testing.assert_array_equal(restored.signal, lw.signal)
        assert restored.sampling_rate == lw.sampling_rate
        assert restored.wave_type == lw.wave_type
        assert restored.source_name == lw.source_name


# ---------------------------------------------------------------------------
# EDF loader smoke test (skipped if test data is absent)
# ---------------------------------------------------------------------------


_EDF_PATH = os.path.join("tests", "test_data", "example.edf")


@pytest.mark.skipif(not os.path.exists(_EDF_PATH), reason="example.edf not bundled")
def test_load_edf_round_trip():
    with open(_EDF_PATH, "rb") as f:
        url = _data_url(f.read(), mime="application/octet-stream")
    lw = load_from_upload(url, "example.edf", wave_type="ECG")
    assert lw.signal.shape[0] > 0
    assert lw.sampling_rate > 0
    assert lw.wave_type == "ECG"


# ---------------------------------------------------------------------------
# Oucru CSV (row-per-second with array-valued signal columns)
# ---------------------------------------------------------------------------


def _make_oucru_ppg_csv(n_rows: int = 3, samples_per_row: int = 100) -> str:
    """Build a small in-memory SmartCare/Oucru-style PPG CSV."""
    rows = []
    for i in range(n_rows):
        # Sinusoid scaled to the typical Oucru PPG range.
        pleth = np.round(
            30000 + 10000 * np.sin(
                2 * np.pi * 1.0 * (np.arange(samples_per_row) / samples_per_row + i)
            )
        ).astype(int).tolist()
        rows.append({
            "timestamp": f"2025-10-11 09:31:0{i+4}.000000+05:45",
            "device_id": "00:11:22:33:44:55",
            "battery": 80,
            "hr": 75,
            "o2": 97,
            "spo2_status": "[0, 0, 0]",
            "pleth": str(pleth),
            "red": str([1] * samples_per_row),
            "ir": str([2] * samples_per_row),
            "perfusion": str([0.5] * samples_per_row),
        })
    df = pd.DataFrame(rows)
    return _csv_url(df)


def _make_oucru_ecg_csv(n_rows: int = 3, samples_per_row: int = 128) -> str:
    rows = []
    for i in range(n_rows):
        ecg = np.round(
            500 * np.sin(
                2 * np.pi * 1.2 * (np.arange(samples_per_row) / samples_per_row + i)
            )
        ).astype(int).tolist()
        rows.append({
            "timestamp": f"2024-07-02 05:23:{15 + i:02d}.676000+00:00",
            "device_id": "C8:80:D1:4B:66:81",
            "battery": 75,
            "ecg": str(ecg),
            "hr": 86,
        })
    df = pd.DataFrame(rows)
    return _csv_url(df)


class TestLoadOucruCsv:
    def test_ppg_detected_and_loaded(self):
        url = _make_oucru_ppg_csv(n_rows=3, samples_per_row=100)
        lw = load_from_upload(url, "SmartCare.csv", wave_type="PPG")
        # 3 rows × 100 samples/s = 300 samples total
        assert lw.signal.shape == (300,)
        assert lw.sampling_rate == 100.0
        assert lw.wave_type == "PPG"

    def test_ecg_detected_and_loaded(self):
        url = _make_oucru_ecg_csv(n_rows=4, samples_per_row=128)
        lw = load_from_upload(url, "Ecg.csv", wave_type="ECG")
        assert lw.signal.shape == (4 * 128,)
        assert lw.sampling_rate == 128.0
        assert lw.wave_type == "ECG"

    def test_timestamps_extracted_per_row(self):
        url = _make_oucru_ppg_csv(n_rows=2, samples_per_row=100)
        lw = load_from_upload(url, "rec.csv", wave_type="PPG")
        # One timestamp per row, not per sample
        assert lw.timestamps is not None
        assert len(lw.timestamps) == 2

    def test_signal_column_override_picks_ir_instead_of_pleth(self):
        url = _make_oucru_ppg_csv(n_rows=2, samples_per_row=100)
        lw = load_from_upload(
            url, "rec.csv", wave_type="PPG", signal_column="ir"
        )
        # `ir` column was filled with the constant 2.
        assert lw.signal.shape == (200,)
        assert np.all(lw.signal == 2.0)

    def test_unsupported_wave_type_column_raises(self):
        # An Oucru-format file that has neither 'ecg' nor 'pleth' should
        # raise rather than fall through to the flat-CSV path.
        df = pd.DataFrame({
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:01"],
            "weird_signal": ["[1,2,3]", "[4,5,6]"],
        })
        url = _csv_url(df)
        # Auto-pick fails for PPG (no pleth/ir/red) → falls through to flat
        # CSV parser which then complains about non-numeric content.  Either
        # error class is acceptable; both are WaveformLoaderError.
        with pytest.raises(WaveformLoaderError):
            load_from_upload(url, "rec.csv", wave_type="PPG", sampling_rate=100)

    def test_short_row_is_padded_not_dropped(self):
        # Row 2 has only 50 samples instead of 100 — the loader should keep
        # the timeline aligned by filling with interpolated values.
        rows = [
            {"timestamp": "2024-01-01 00:00:00",
             "pleth": str(list(range(100)))},
            {"timestamp": "2024-01-01 00:00:01",
             "pleth": str(list(range(50)))},
        ]
        df = pd.DataFrame(rows)
        url = _csv_url(df)
        lw = load_from_upload(url, "rec.csv", wave_type="PPG")
        # 2 rows × 100 samples (modal length) = 200, never 50+100=150
        assert lw.signal.shape == (200,)
        assert np.all(np.isfinite(lw.signal))

    def test_introspect_lists_array_and_numeric_columns(self):
        # Oucru-style file: pleth is array; hr is numeric scalar.
        df = pd.DataFrame({
            "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:01"],
            "hr": [70, 72],
            "pleth": [str(list(range(100))), str(list(range(100)))],
            "ir": [str([1]*100), str([1]*100)],
        })
        url = _csv_url(df)
        cands = introspect_columns(url, "rec.csv", wave_type="PPG")
        names = [c.name for c in cands]
        assert "pleth" in names
        assert "ir" in names
        assert "hr" in names

    def test_introspect_marks_preferred_first(self):
        # 'pleth' is the canonical PPG signal so it must appear before
        # 'ir' (also preferred) regardless of source column order.
        df = pd.DataFrame({
            "timestamp": ["2024-01-01 00:00:00"],
            "ir": [str([1] * 100)],
            "pleth": [str([2] * 100)],
        })
        cands = introspect_columns(_csv_url(df), "rec.csv", wave_type="PPG")
        names = [c.name for c in cands]
        assert names.index("pleth") < names.index("ir")
        assert cands[0].preferred is True

    def test_introspect_returns_empty_for_edf(self):
        # Binary formats — the loader's reader exposes a single channel; the
        # UI has nothing to pick, so we return [].
        assert introspect_columns("data:application/octet-stream;base64,Zm9v", "rec.edf", "ECG") == []

    def test_introspect_returns_empty_for_malformed_upload(self):
        assert introspect_columns("not-a-data-url", "rec.csv", "PPG") == []

    def test_bracketless_array_cell_parses(self):
        # The ECG file's acc_x/y/z columns are bare comma-separated lists
        # (no surrounding brackets).  Our cell parser must accept both.
        rows = [
            {"timestamp": "2024-01-01 00:00:00",
             "ecg": "1,2,3,4,5,6,7,8,9,10"},
            {"timestamp": "2024-01-01 00:00:01",
             "ecg": "11,12,13,14,15,16,17,18,19,20"},
        ]
        df = pd.DataFrame(rows)
        url = _csv_url(df)
        lw = load_from_upload(url, "rec.csv", wave_type="ECG")
        assert lw.signal.shape == (20,)
        assert lw.sampling_rate == 10.0
