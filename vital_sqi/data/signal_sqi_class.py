import numpy as np
import pandas as pd
import json
import logging
from vital_sqi.common.band_filter import BandpassFilter
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.resource import sqi_dict, rule_dict
from vital_sqi.rule import Rule, RuleSet

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SignalSQI:
    """
    A class representing a signal with its associated Signal Quality Index (SQI) values,
    rules, and rule set for quality analysis.
    """

    def __init__(
        self,
        wave_type=None,
        signals=None,
        sampling_rate=None,
        start_datetime=None,
        info=None,
        sqis=None,
        rules=None,
        ruleset=None,
    ):
        self.wave_type = wave_type
        self.signals = signals
        self.sampling_rate = sampling_rate
        self.start_datetime = (
            start_datetime or pd.Timestamp.now()
        )  # Set to current datetime if None

        # Explicitly handle the `info` attribute
        if isinstance(info, (pd.DataFrame, list, dict)):
            self.info = info
        else:
            self.info = None

        # Load sqis from sqi_dict.json if not provided
        self.sqis = sqis if sqis is not None else sqi_dict

        # Load rules from rule_dict.json if not provided
        self.rules = (
            rules if rules is not None else self._load_rules_from_dict(rule_dict)
        )

        # Load the default rule set from the first rule in rule_dict.json if not provided
        self.ruleset = ruleset if ruleset is not None else self._load_default_rule_set()

    def __setattr__(self, name, value):
        if name == "wave_type":
            if value not in ("ECG", "PPG"):
                raise ValueError("Expected wave_type to be either 'ECG' or 'PPG'.")
        elif name == "signals":
            if not isinstance(value, pd.DataFrame):
                raise ValueError(
                    "Expected signals as a pd.DataFrame with one channel per column."
                )
        elif name == "sampling_rate":
            if not np.isreal(value):
                raise ValueError("Expected a numeric value for sampling_rate.")
        elif name == "info":
            if value is not None and not isinstance(value, (list, dict, pd.DataFrame)):
                raise ValueError("Expected info as a list, dict, or pd.DataFrame.")
        elif name == "sqis":
            if value is not None and not isinstance(value, (pd.DataFrame, list, dict)):
                raise ValueError(
                    "Expected sqis as a pd.DataFrame, list of DataFrames, or None."
                )
        elif name == "rules":
            if value is not None and not isinstance(value, dict):
                raise ValueError("Expected rules as a dictionary of Rule objects.")
        elif name == "ruleset":
            if value is not None and not isinstance(value, RuleSet):
                raise ValueError("Expected ruleset as a RuleSet object.")
        super().__setattr__(name, value)

    def _load_rules_from_dict(self, rule_data):
        """Initialize Rule objects from the loaded rule dictionary."""
        return {k: Rule(k).load_def(rule_data) for k in rule_data.keys()}

    # def _load_default_rule_set(self):
    #     """Initialize the default RuleSet using the first rule in the rule_dict."""
    #     if self.rules:
    #         first_rule_key = next(iter(self.rules))
    #         rule_set = RuleSet({first_rule_key: self.rules[first_rule_key]})
    #         return rule_set
    #     else:
    #         logging.error("Cannot initialize RuleSet: rules not loaded properly.")
    #         return None

    def _load_default_rules(self):
        """
        Loads the default rules for the SQI from `rule_dict`.
        """
        if isinstance(rule_dict, dict):
            try:
                self.rules = {name: Rule(name) for name in rule_dict.keys()}
            except Exception as e:
                logging.error(f"Failed to initialize default rules: {e}")
                self.rules = None
        else:
            logging.error("rule_dict must be a dictionary.")
            self.rules = None

    def _load_default_rule_set(self):
        """
        Initialize the default RuleSet using rules with consecutive numbering
        starting from 1, or raise a warning if rules are missing.
        """
        self._load_default_rules()
        if self.rules:
            ordered_rules = {
                i + 1: rule for i, (key, rule) in enumerate(self.rules.items())
            }
            try:
                rule_set = RuleSet(ordered_rules)
                return rule_set
            except ValueError as e:
                logging.error(f"Error in initializing RuleSet: {e}")
                raise ValueError(
                    "Failed to initialize RuleSet due to non-consecutive rules."
                )
        else:
            logging.error("Cannot initialize RuleSet: rules not loaded properly.")
            return None

    def update_info(self, info):
        """
        Update the info attribute.

        Parameters
        ----------
        info : list, dict, or pd.DataFrame
            Metadata information.

        Returns
        -------
        SignalSQI
            Updated SignalSQI object.
        """
        self.info = info
        return self

    def update_signals(self, signals):
        """
        Update the signals attribute.

        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame with signal data, where each column represents a channel.

        Returns
        -------
        SignalSQI
            Updated SignalSQI object.
        """
        self.signals = signals
        return self

    def update_sampling_rate(self, sampling_rate):
        """
        Update the sampling rate.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate of the signal.

        Returns
        -------
        SignalSQI
            Updated SignalSQI object.
        """
        self.sampling_rate = sampling_rate
        return self

    def update_start_datetime(self, start_datetime):
        """
        Update the start datetime.

        Parameters
        ----------
        start_datetime : datetime
            Starting datetime of the signal.

        Returns
        -------
        SignalSQI
            Updated SignalSQI object.
        """
        self.start_datetime = start_datetime
        return self
