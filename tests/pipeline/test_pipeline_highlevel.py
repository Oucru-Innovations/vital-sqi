import os
import tempfile
import pytest
from vital_sqi.pipeline.pipeline_highlevel import (
    get_ppg_sqis,
    get_ecg_sqis,
    get_qualified_ppg,
    get_qualified_ecg,
)
from vital_sqi.data.signal_sqi_class import SignalSQI
from unittest.mock import patch
import pandas as pd
import json


class TestGetPPGSQIs:
    def test_on_valid_ppg_file(self):
        """Test get_ppg_sqis with a valid PPG file."""
        file_in = os.path.abspath("tests/test_data/ppg_smartcare.csv")
        sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
        rule_dict_filename = os.path.abspath("tests/test_data/rule_dict_test.json")
        ruleset_order = {2: "skewness_1", 1: "perfusion"}
        output_dir = tempfile.gettempdir()
        # output_dir = "D:\Workspace\Oucru\\vital_sqi\outdir"

        # Call the function under test
        segments, signal_obj = get_ppg_sqis(
            file_name=file_in,
            sqi_dict_filename=sqi_dict,
            signal_idx=6,
            timestamp_idx=0,
            # file_type="edf",  # File type explicitly defined as in the example
            duration=30,  # Duration parameter passed
            # rule_dict_filename=rule_dict_filename,
            # ruleset_order=ruleset_order,
            # output_dir=output_dir,
        )
        assert isinstance(segments, list), "Segments should be a list."
        assert isinstance(
            signal_obj, SignalSQI
        ), "Returned object should be of type SignalSQI."
        assert signal_obj.sqis is not None, "SQIs should not be None."

        signal_obj = get_qualified_ppg(
            file_name=file_in,
            sqi_dict_filename=sqi_dict,
            signal_idx=6,
            timestamp_idx=0,
            # file_type="edf",  # File type explicitly defined as in the example
            duration=30,  # Duration parameter passed
            rule_dict_filename=rule_dict_filename,
            ruleset_order=ruleset_order,
            output_dir=output_dir,
        )
        # assert isinstance(segments, list), "Segments should be a list."
        assert isinstance(
            signal_obj, SignalSQI
        ), "Returned object should be of type SignalSQI."
        assert signal_obj.sqis is not None, "SQIs should not be None."

    def test_on_missing_file(self):
        """Test get_ppg_sqis with a missing PPG file."""
        sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
        with pytest.raises(FileNotFoundError):
            get_ppg_sqis(
                "non_existent_file.csv",
                timestamp_idx=["TIMESTAMP_MS"],
                signal_idx=["PLETH"],
                sqi_dict_filename=sqi_dict,
            )

    def test_on_empty_ppg_file(self):
        """Test get_ppg_sqis with an empty PPG file."""
        empty_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        empty_file.close()
        sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")

        with pytest.raises(ValueError):
            get_ppg_sqis(
                empty_file.name,
                timestamp_idx=["TIMESTAMP_MS"],
                signal_idx=["PLETH"],
                sqi_dict_filename=sqi_dict,
            )

        os.unlink(empty_file.name)


# class TestGetQualifiedPPG:
#     @patch("vital_sqi.pipeline.pipeline_functions.extract_sqi")
#     def test_on_valid_ppg_with_classification(self, mock_extract_sqi):
#         """Test get_qualified_ppg with valid classification."""
#         # Mock `extract_sqi` to return expected SQIs
#         mock_extract_sqi.return_value = pd.DataFrame({
#             "skewness_1": [0.1, -0.1, 0.2],
#             "entropy": [0.5, 0.3, -0.2],
#             "perfusion": [3.0, 2.8, 4.0],
#             "decision": ["accept", "reject", "accept"]
#         })

#         # Example-based input files and parameters
#         file_in = os.path.abspath("tests/test_data/example.edf")
#         sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
#         rule_dict_filename = os.path.abspath("tests/test_data/rule_dict_test.json")
#         ruleset_order = {3: "skewness_1", 2: "entropy", 1: "perfusion"}
#         output_dir = tempfile.gettempdir()

#         # Call the function under test
#         signal_obj = get_qualified_ppg(
#             file_name=file_in,
#             sqi_dict_filename=sqi_dict,
#             signal_idx=6,
#             timestamp_idx=0,
#             # file_type="edf",  # File type explicitly defined as in the example
#             duration=30,      # Duration parameter passed
#             rule_dict_filename=rule_dict_filename,
#             ruleset_order=ruleset_order,
#             output_dir=output_dir,
#         )

#         # Assertions to validate results
#         assert signal_obj is not None
#         assert "decision" in signal_obj.columns
#         assert all(signal_obj["decision"].isin(["accept", "reject"]))


#     def test_on_invalid_rule_dict(self):
#         """Test get_qualified_ppg with an invalid rule dictionary."""
#         file_in = os.path.abspath("tests/test_data/ppg_smartcare.csv")
#         sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
#         rule_dict_filename = "invalid_rule_dict.json"
#         ruleset_order = {3: "skewness_1", 2: "entropy", 1: "perfusion"}
#         output_dir = tempfile.gettempdir()

#         with pytest.raises(FileNotFoundError):
#             get_qualified_ppg(
#                 file_in,
#                 sqi_dict_filename=sqi_dict,
#                 signal_idx=["PLETH"],
#                 timestamp_idx=["TIMESTAMP_MS"],
#                 rule_dict_filename=rule_dict_filename,
#                 ruleset_order=ruleset_order,
#                 output_dir=output_dir,
#                 save_image=True,
#             )


class TestGetECGSQIs:
    def test_on_valid_ecg_file(self):
        """Test get_ecg_sqis with a valid ECG file."""
        file_in = os.path.abspath("tests/test_data/example.edf")
        sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
        segments, signal_sqi_obj = get_ecg_sqis(file_in, sqi_dict, "edf")

        assert isinstance(segments, list), "Segments should be a list."
        assert isinstance(
            signal_sqi_obj, SignalSQI
        ), "Returned object should be of type SignalSQI."
        assert signal_sqi_obj.sqis is not None, "SQIs should not be None."

        sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
        rule_dict_filename = os.path.abspath("tests/test_data/rule_dict_test.json")
        ruleset_order = {2: "skewness_1", 1: "perfusion"}
        output_dir = tempfile.gettempdir()
        # output_dir = "D:\Workspace\Oucru\\vital_sqi\outdir"
        signal_obj = get_qualified_ecg(
            file_name=file_in,
            file_type="edf",
            sqi_dict_filename=sqi_dict,
            # signal_idx=6,
            # timestamp_idx=0,
            # file_type="edf",  # File type explicitly defined as in the example
            duration=30,  # Duration parameter passed
            rule_dict_filename=rule_dict_filename,
            ruleset_order=ruleset_order,
            output_dir=output_dir,
        )
        # assert isinstance(segments, list), "Segments should be a list."
        assert isinstance(
            signal_obj, SignalSQI
        ), "Returned object should be of type SignalSQI."
        assert signal_obj.sqis is not None, "SQIs should not be None."

    # def test_on_empty_ecg_file(self):
    #     """Test get_ecg_sqis with an empty ECG file."""
    #     empty_file = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
    #     empty_file.close()
    #     sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")

    #     with pytest.raises(ValueError):
    #         get_ecg_sqis(empty_file.name, sqi_dict, "edf")

    #     os.unlink(empty_file.name)


# class TestGetQualifiedECG:
#     def test_on_valid_ecg_with_classification(self):
#         """Test get_qualified_ecg with valid ECG data and classification."""
#         file_in = os.path.abspath("tests/test_data/example.edf")
#         sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
#         rule_dict_filename = os.path.abspath("tests/test_data/rule_dict_test.json")
#         ruleset_order = {3: "skewness_1", 2: "entropy", 1: "perfusion"}
#         output_dir = tempfile.gettempdir()

#         signal_obj = get_qualified_ecg(
#             file_name=file_in,
#             sqi_dict_filename=sqi_dict,
#             file_type="edf",
#             duration=30,
#             rule_dict_filename=rule_dict_filename,
#             ruleset_order=ruleset_order,
#             output_dir=output_dir,
#         )

#         assert isinstance(signal_obj, SignalSQI), "Signal object should be of type SignalSQI."
#         assert os.path.isdir(os.path.join(output_dir, "accept", "img")), "Accepted segments folder should exist."
#         assert os.path.isdir(os.path.join(output_dir, "reject", "img")), "Rejected segments folder should exist."

#     def test_on_missing_ecg_file(self):
#         """Test get_qualified_ecg with a missing ECG file."""
#         sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
#         rule_dict_filename = os.path.abspath("tests/test_data/rule_dict_test.json")
#         ruleset_order = {3: "skewness_1", 2: "entropy", 1: "perfusion"}

#         with pytest.raises(FileNotFoundError):
#             get_qualified_ecg(
#                 file_name="non_existent.edf",
#                 sqi_dict_filename=sqi_dict,
#                 file_type="edf",
#                 duration=30,
#                 rule_dict_filename=rule_dict_filename,
#                 ruleset_order=ruleset_order,
#             )
