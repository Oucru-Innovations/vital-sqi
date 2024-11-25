import logging
from pyedflib import highlevel
from wfdb import rdsamp, wrsamp
import numpy as np
import pandas as pd
import datetime as dt
import os
import glob
from vital_sqi.common import utils
from vital_sqi.common.utils import generate_timestamp
from vital_sqi.data.signal_sqi_class import SignalSQI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def ECG_reader(
    file_name,
    file_type,
    channel_num=None,
    channel_name=None,
    sampling_rate=None,
    start_datetime=None,
):
    """
    Reads ECG data from a specified file type and returns a SignalSQI object.

    Parameters
    ----------
    file_name : str
        Path to ECG file.
    file_type : str
        Supported types include 'edf', 'mit' or 'csv'.
    channel_num : list, optional
        List of channel ids to read, starting from 0.
    channel_name : list, optional
        List of channel names to read.
    sampling_rate : int or float, optional
        Sampling rate of the signal.
    start_datetime : str, optional
        Start datetime in '%Y-%m-%d %H:%M:%S.%f' format.

    Returns
    -------
    SignalSQI
        SignalSQI object.
    """
    try:
        if file_type == "mit":
            assert glob.glob(file_name + ".*"), "Files not found"
        else:
            assert os.path.isfile(file_name), "File not found"

        assert file_type in [
            "edf",
            "mit",
            "csv",
        ], "Only edf, mit, and csv are supported."
        assert (
            isinstance(channel_num, list) or channel_num is None
        ), "Channel num must be a list or None"
        assert (
            isinstance(channel_name, list) or channel_name is None
        ), "Channel name must be a list or None"
        assert not (
            channel_name and channel_num
        ), "Specify either channel name or channel index(s) or None"
        assert (
            isinstance(start_datetime, str) or start_datetime is None
        ), "Start datetime must be None or a string"
        assert (
            isinstance(sampling_rate, (int, float)) or sampling_rate is None
        ), "Sampling rate must be a number or None"

        if isinstance(sampling_rate, float):
            sampling_rate = round(sampling_rate)
        if start_datetime:
            start_datetime = utils.parse_datetime(start_datetime)

        if file_type == "edf":
            signals, signal_headers, header = highlevel.read_edf(
                edf_file=file_name, ch_nrs=channel_num, ch_names=channel_name
            )
            sampling_rate = sampling_rate or signal_headers[0].get("sample_rate")
            if sampling_rate is None:
                raise ValueError("Sampling rate could not be inferred.")
            start_datetime = start_datetime or header.get("startdate")
            signals = pd.DataFrame(signals.T)
            timestamps = generate_timestamp(start_datetime, sampling_rate, len(signals))
            signals.insert(0, "timestamps", timestamps)
            info = [header, signal_headers]
            return SignalSQI(
                signals=signals,
                wave_type="ECG",
                start_datetime=start_datetime,
                sampling_rate=sampling_rate,
                info=info,
            )

        elif file_type == "mit":
            signals, info = rdsamp(
                file_name,
                channels=channel_num,
                channel_names=channel_name,
                warn_empty=True,
            )
            sampling_rate = sampling_rate or info.get("fs")
            if sampling_rate is None:
                raise ValueError("Sampling rate could not be inferred.")
            if not start_datetime:
                date, time = info.get("base_date"), info.get("base_time")
                start_datetime = (
                    dt.datetime.combine(date, time) if date and time else None
                )
            timestamps = generate_timestamp(start_datetime, sampling_rate, len(signals))
            signals = pd.DataFrame(signals)
            signals["timestamps"] = timestamps
            return SignalSQI(
                signals=signals, wave_type="ECG", sampling_rate=sampling_rate, info=info
            )

        elif file_type == "csv":
            signals = pd.read_csv(file_name, usecols=channel_name or channel_num)
            timestamps = (
                signals.iloc[:, 0].apply(pd.Timestamp)
                if not start_datetime
                else generate_timestamp(start_datetime, sampling_rate, len(signals))
            )
            if sampling_rate is None:
                raise ValueError("Sampling rate could not be inferred.")
            signals["timestamps"] = timestamps
            return SignalSQI(
                signals=signals,
                wave_type="ECG",
                start_datetime=start_datetime,
                sampling_rate=sampling_rate,
            )

    except Exception as e:
        logging.error(f"Failed to read ECG file: {e}")
        raise


def ECG_writer(signal_sqi, file_name, file_type, info=None):
    """
    Writes the SignalSQI object to a file.

    Parameters
    ----------
    signal_sqi : SignalSQI
        SignalSQI object containing signals, sampling rate, and sqi.
    file_name : str
        Name of the file to write, with extension.
    file_type : str
        Type of file ('edf', 'mit', 'csv').
    info : list or dict, optional
        Additional header information.
    """
    try:
        signals = signal_sqi.signals.drop(columns="timestamps").to_numpy()
        sampling_rate = signal_sqi.sampling_rate
        start_datetime = signal_sqi.start_datetime

        if file_type == "edf":
            if info:
                signal_headers, header = info[1], info[0]
                highlevel.write_edf(
                    file_name, signals.T, signal_headers, header, file_type=-1
                )
            else:
                highlevel.write_edf_quick(file_name, signals.T, sampling_rate)

        elif file_type == "mit":
            if not info:
                raise ValueError("Header dict needed for MIT format")
            wrsamp(
                record_name=file_name.split("/")[-1],
                fs=sampling_rate,
                units=info["units"],
                sig_name=info["sig_name"],
                p_signal=signals,
                base_date=info["base_date"],
                base_time=info["base_time"],
                comments=info["comments"],
                write_dir="/".join(file_name.split("/")[:-1]),
            )

        elif file_type == "csv":
            timestamps = generate_timestamp(start_datetime, sampling_rate, len(signals))
            signals = pd.DataFrame(
                np.hstack((np.array(timestamps).reshape(-1, 1), signals))
            )
            signals.to_csv(file_name, index=False)

        return True  # Ensure success is indicated

    except Exception as e:
        logging.error(f"Failed to write ECG file: {e}")
        raise


def PPG_reader(
    file_name,
    signal_idx,
    timestamp_idx,
    info_idx=[],
    timestamp_unit="ms",
    sampling_rate=None,
    start_datetime=None,
):
    """
    Reads PPG data from a CSV file and returns a SignalSQI object.

    This function extracts PPG signal data from a CSV file. It converts the
    timestamp column to `pd.Timestamp` format, based on the provided `timestamp_unit`,
    and generates timestamps accordingly if `start_datetime` is provided. If the sampling
    rate is not specified, it attempts to infer it based on the timestamps.

    Parameters
    ----------
    file_name : str
        Path to the PPG file (CSV format).
    signal_idx : list
        List of indices or names indicating the columns with PPG signal data.
    timestamp_idx : list
        List containing the index or name of the column with timestamp data.
    info_idx : list, optional
        List of indices or names of columns with additional information. Default is an empty list.
    timestamp_unit : str, optional
        Unit of timestamp in the file. Accepts "ms" (milliseconds) or "s" (seconds). Default is "ms".
    sampling_rate : int or float, optional
        Sampling rate of the PPG signal. If None, it will be inferred from the timestamps. Default is None.
    start_datetime : str, optional
        Start datetime in '%Y-%m-%d %H:%M:%S.%f' format. If None, the current time will be used.

    Returns
    -------
    SignalSQI
        SignalSQI object containing the PPG signals, metadata, and timestamps.

    Raises
    ------
    Exception
        If there are issues reading the file or interpreting the timestamps.
    """
    try:
        # Validate timestamp_unit
        valid_units = ["ms", "s"]
        if timestamp_unit not in valid_units:
            raise ValueError(
                "Timestamp unit must be either 'ms' (milliseconds) or 's' (seconds)."
            )

        info_idx = info_idx or []
        if isinstance(signal_idx, int):
            signal_idx = [signal_idx]

        # Ensure info_idx is a list
        info_idx = info_idx or []

        # Combine all columns to read
        if type(timestamp_idx) is not list:
            timestamp_idx = [timestamp_idx]
        cols = timestamp_idx + signal_idx + info_idx

        tmp = pd.read_csv(
            file_name, usecols=cols, skipinitialspace=True, skip_blank_lines=True
        )

        # Process timestamps
        timestamps = tmp.iloc[:, 0]
        if timestamp_unit == "ms":
            timestamps = pd.to_datetime(timestamps, unit="ms")
        elif timestamp_unit == "s":
            timestamps = pd.to_datetime(timestamps, unit="s")

        # Adjust timestamps if start_datetime is provided
        if start_datetime:
            start_datetime = pd.Timestamp(start_datetime)
            timestamps += start_datetime - timestamps.iloc[0]

        # Infer sampling rate if not provided
        if sampling_rate is None:
            diffs = timestamps.diff().dt.total_seconds()
            sampling_rate = 1 / diffs.median()

        # Extract signal data and additional info
        signal_data = tmp.iloc[:, 1 : 1 + len(signal_idx)]
        additional_info = tmp.iloc[:, 1 + len(signal_idx) :]

        signal_df = pd.concat([timestamps, signal_data, additional_info], axis=1)
        # Return a SignalSQI object (or similar custom object)
        return SignalSQI(
            signals=signal_df,
            wave_type="PPG",
            # timestamps=timestamps,
            info=additional_info,
            sampling_rate=sampling_rate,
        )

    except Exception as e:
        logging.error(f"Failed to read PPG file: {e}")
        raise


def PPG_writer(signal_sqi, file_name, file_type="csv"):
    """
    Writes PPG SignalSQI data to a specified file format.

    This function exports the `SignalSQI` object containing PPG signal data to either a CSV
    or Excel file format. The output file includes timestamps and PPG signal values.

    Parameters
    ----------
    signal_sqi : SignalSQI
        SignalSQI object containing PPG signals, timestamps, and metadata.
    file_name : str
        Absolute path for the output file, including the file extension.
    file_type : str, optional
        Type of file to write. Options are "csv" (default) or "xlsx" for Excel format.

    Returns
    -------
    bool
        True if the file was successfully written, False otherwise.

    Raises
    ------
    Exception
        If there are issues writing the file or converting data formats.
    """
    try:
        # Generate timestamps for the signal based on start time and sampling rate
        timestamps = generate_timestamp(
            start_datetime=signal_sqi.start_datetime,
            sampling_rate=signal_sqi.sampling_rate,
            signal_length=len(signal_sqi.signals),
        )
        # signals = pd.DataFrame(
        #     {"time": timestamps, "pleth": np.array(signal_sqi.signals).reshape(-1)}
        # )
        signals = pd.DataFrame(
            {
                "time": np.array(signal_sqi.signals.iloc[:, 0]),
                "pleth": np.array(signal_sqi.signals.iloc[:, 1]),
            }
        )

        if file_type == "csv":
            signals.to_csv(file_name, index=False)
        elif file_type == "xlsx":
            signals.to_excel(file_name, index=False)
        else:
            logging.error(f"Unsupported file type: {file_type}")
            return False

        return os.path.isfile(file_name)

    except Exception as e:
        logging.error(f"Failed to write PPG file: {e}")
        raise
