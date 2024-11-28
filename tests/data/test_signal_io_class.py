import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
from vital_sqi.data.signal_sqi_class import SignalSQI
from vital_sqi.rule import Rule, RuleSet
from vitalDSP.utils.synthesize_data import generate_ecg_signal


class TestSignalSQI:
    @pytest.fixture
    def valid_signals(self):
        return pd.DataFrame(
            {"channel1": np.random.randn(100), "channel2": np.random.randn(100)}
        )

    @pytest.fixture
    def valid_rule_dict(self):
        return {
            "rule1": {
                "def": "sqi > 0.8",
                "boundaries": [0, 1],
                "labels": ["bad", "good"],
            },
            "rule2": {
                "def": "sqi < 0.5",
                "boundaries": [0, 1],
                "labels": ["bad", "good"],
            },
        }

    @pytest.fixture
    def mock_rule(self):
        with patch("vital_sqi.rule.Rule") as mock_rule:
            mock_rule.return_value = MagicMock()
            yield mock_rule

    @pytest.fixture
    def mock_ruleset(self):
        with patch("vital_sqi.rule.RuleSet") as mock_ruleset:
            mock_ruleset.return_value = MagicMock()
            yield mock_ruleset

    @pytest.fixture
    def valid_info(self):
        return {"subject": "Test", "date": "2024-01-01"}

    def test_initialization_with_defaults(self, mock_rule, mock_ruleset):
        """Test default initialization."""
        with patch("vital_sqi.resource.rule_dict", {"rule1": {}, "rule2": {}}), patch(
            "vital_sqi.resource.sqi_dict", {"sqi1": {}, "sqi2": {}}
        ):
            sfecg = 256
            N = 100
            Anoise = 0.05
            hrmean = 70
            ecg_signal = generate_ecg_signal(
                sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
            )
            sqi = SignalSQI(signals=pd.DataFrame({"channel": ecg_signal}))
            assert sqi.wave_type == "ECG"
            assert isinstance(sqi.start_datetime, pd.Timestamp)
            assert sqi.rules is not None
            assert isinstance(sqi.signals, pd.DataFrame)

    def test_wave_type_validation(self):
        """Test wave_type attribute validation."""
        with pytest.raises(
            ValueError, match="Expected wave_type to be either 'ECG' or 'PPG'."
        ):
            sfecg = 256
            N = 100
            Anoise = 0.05
            hrmean = 70
            ecg_signal = generate_ecg_signal(
                sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
            )
            SignalSQI(
                signals=pd.DataFrame({"channel": ecg_signal}), wave_type="Invalid"
            )

    def test_signals_validation(self):
        """Test signals attribute validation."""
        with pytest.raises(ValueError, match="Expected signals as a pd.DataFrame"):
            SignalSQI(signals="Invalid signals")

    def test_sampling_rate_validation(self):
        """Test sampling_rate attribute validation."""
        with pytest.raises(
            ValueError, match="Expected a numeric value for sampling_rate."
        ):
            sfecg = 256
            N = 100
            Anoise = 0.05
            hrmean = 70
            ecg_signal = generate_ecg_signal(
                sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
            )
            SignalSQI(
                signals=pd.DataFrame({"channel": ecg_signal}), sampling_rate="Invalid"
            )

    # def test_info_validation(self):
    #     """Test info attribute validation."""
    #     with pytest.raises(ValueError, match="Expected info as a list, dict, or pd.DataFrame."):
    #         sfecg = 256
    #         N = 100
    #         Anoise = 0.05
    #         hrmean = 70
    #         ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
    #         SignalSQI(signals=pd.DataFrame({'channel':ecg_signal}),info="Invalid info")

    def test_sqis_validation(self):
        """Test sqis attribute validation."""
        with pytest.raises(
            ValueError,
            match="Expected sqis as a pd.DataFrame, list of DataFrames, or None.",
        ):
            sfecg = 256
            N = 100
            Anoise = 0.05
            hrmean = 70
            ecg_signal = generate_ecg_signal(
                sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
            )
            SignalSQI(
                signals=pd.DataFrame({"channel": ecg_signal}), sqis="Invalid sqis"
            )

    def test_rules_validation(self):
        """Test rules attribute validation."""
        with pytest.raises(
            ValueError, match="Expected rules as a dictionary of Rule objects."
        ):
            sfecg = 256
            N = 100
            Anoise = 0.05
            hrmean = 70
            ecg_signal = generate_ecg_signal(
                sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
            )
            SignalSQI(
                signals=pd.DataFrame({"channel": ecg_signal}), rules="Invalid rules"
            )

    def test_ruleset_validation(self):
        """Test ruleset attribute validation."""
        with pytest.raises(ValueError, match="Expected ruleset as a RuleSet object."):
            sfecg = 256
            N = 100
            Anoise = 0.05
            hrmean = 70
            ecg_signal = generate_ecg_signal(
                sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
            )
            SignalSQI(
                signals=pd.DataFrame({"channel": ecg_signal}), ruleset="Invalid ruleset"
            )

    def test_update_info(self, valid_info):
        """Test updating info attribute."""
        sfecg = 256
        N = 100
        Anoise = 0.05
        hrmean = 70
        ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
        sqi = SignalSQI(signals=pd.DataFrame({"channel": ecg_signal}))
        sqi.update_info(valid_info)
        assert sqi.info == valid_info

    def test_update_signals(self, valid_signals):
        """Test updating signals attribute."""
        sfecg = 256
        N = 100
        Anoise = 0.05
        hrmean = 70
        ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
        sqi = SignalSQI(signals=pd.DataFrame({"channel": ecg_signal}))
        sqi.update_signals(valid_signals)
        pd.testing.assert_frame_equal(sqi.signals, valid_signals)

    def test_update_sampling_rate(self):
        """Test updating sampling_rate attribute."""
        sfecg = 256
        N = 100
        Anoise = 0.05
        hrmean = 70
        ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
        sqi = SignalSQI(signals=pd.DataFrame({"channel": ecg_signal}))
        sqi.update_sampling_rate(100.0)
        assert sqi.sampling_rate == 100.0

    def test_update_start_datetime(self):
        """Test updating start_datetime attribute."""
        new_datetime = datetime(2024, 1, 1)
        sfecg = 256
        N = 100
        Anoise = 0.05
        hrmean = 70
        ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
        sqi = SignalSQI(signals=pd.DataFrame({"channel": ecg_signal}))
        sqi.update_start_datetime(new_datetime)
        assert sqi.start_datetime == new_datetime

    # def test_load_rules_from_dict(self, valid_rule_dict, mock_rule):
    #     """Test _load_rules_from_dict method."""
    #     sfecg = 256
    #     N = 100
    #     Anoise = 0.05
    #     hrmean = 70
    #     ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
    #     sqi = SignalSQI(signals=pd.DataFrame({'channel':ecg_signal}))
    #     with patch("vital_sqi.resource.rule_dict", valid_rule_dict):
    #         rules = sqi._load_rules_from_dict(valid_rule_dict)
    #         assert isinstance(rules, dict)
    #         assert "rule1" in rules
    #         assert "rule2" in rules

    # def test_load_default_rule_set(self, mock_ruleset, valid_rule_dict):
    #     """Test _load_default_rule_set method."""
    #     with patch("vital_sqi.resource.rule_dict", valid_rule_dict):
    #         sfecg = 256
    #         N = 100
    #         Anoise = 0.05
    #         hrmean = 70
    #         ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
    #         sqi = SignalSQI(signals=pd.DataFrame({'channel':ecg_signal}))
    #         sqi._load_default_rules()
    #         ruleset = sqi._load_default_rule_set()
    #         assert isinstance(ruleset, MagicMock)

    # def test_load_default_rules(self, mock_rule, valid_rule_dict):
    #     """Test _load_default_rules method."""
    #     with patch("vital_sqi.resource.rule_dict", valid_rule_dict):
    #         sfecg = 256
    #         N = 100
    #         Anoise = 0.05
    #         hrmean = 70
    #         ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
    #         sqi = SignalSQI(signals=pd.DataFrame({'channel':ecg_signal}))
    #         sqi._load_default_rules()
    #         assert sqi.rules is not None
    #         assert len(sqi.rules) == len(valid_rule_dict)
    #         mock_rule.assert_called()

    # def test_load_default_rule_set_error(self):
    #     """Test _load_default_rule_set method with error."""
    #     sqi = SignalSQI()
    #     with patch("vital_sqi.resource.rule_dict", {}), pytest.raises(ValueError, match="Failed to initialize RuleSet"):
    #         sqi._load_default_rule_set()
