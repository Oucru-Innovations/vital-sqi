import os
import pandas as pd
import warnings
from vital_sqi.preprocess.segment_split import split_segment, save_segment
from vital_sqi.pipeline.pipeline_functions import (
    extract_sqi,
    classify_segments,
    get_reject_segments,
    get_decision_segments,
)
from vital_sqi.data.signal_io import PPG_reader, ECG_reader
import json

warnings.filterwarnings("ignore")


def get_ppg_sqis(
    file_name,
    signal_idx,
    timestamp_idx,
    sqi_dict_filename,
    info_idx=None,
    timestamp_unit="ms",
    sampling_rate=None,
    start_datetime=None,
    split_type=0,
    duration=30,
    overlapping=None,
    peak_detector=6,
    delete_signal=True,
):
    """
    Computes SQIs for PPG segments and returns the segments along with the SQIs.

    Parameters
    ----------
    file_name : str
        Path to the PPG file.
    sqi_dict_filename : str
        Path to the SQI dictionary.
    signal_idx : int
        Index of the signal column in the file.
    timestamp_idx : int
        Index of the timestamp column in the file.
    info_idx : list, optional
        List of indices for additional information columns (default is None).
    timestamp_unit : str, optional
        Time unit for the timestamps (default is 'ms').
    sampling_rate : int, optional
        Sampling rate of the signal (default is None).
    start_datetime : datetime, optional
        Start datetime of the signal (default is None).
    split_type : int, optional
        Type of segment split (default is 0).
    duration : int, optional
        Duration of each segment in seconds (default is 30).
    overlapping : float, optional
        Overlapping ratio between segments (default is None).
    peak_detector : int, optional
        Method for peak detection (default is 7).
    delete_signal : bool, optional
        Whether to delete original signals after segmentation (default is True).

    Returns
    -------
    tuple
        Segments and signal object with SQIs.
    """
    info_idx = info_idx or []
    signal_obj = PPG_reader(
        file_name=file_name,
        signal_idx=signal_idx,
        timestamp_idx=timestamp_idx,
        info_idx=info_idx,
        timestamp_unit=timestamp_unit,
        sampling_rate=sampling_rate,
        start_datetime=start_datetime,
    )

    if info_idx:
        signal_obj.signals = pd.concat([signal_obj.signals, signal_obj.info], axis=1)

    segments, milestones = split_segment(
        signal_obj.signals,
        sampling_rate=signal_obj.sampling_rate,
        split_type=split_type,
        duration=duration,
        overlapping=overlapping,
        peak_detector=peak_detector,
        wave_type="PPG",
    )

    if delete_signal:
        signal_obj.signals = pd.DataFrame()

    signal_obj.sqis = [
        extract_sqi(segments, milestones, sqi_dict_filename, wave_type="PPG")
    ]
    return segments, signal_obj


def get_qualified_ppg(
    file_name,
    sqi_dict_filename,
    signal_idx,
    timestamp_idx,
    rule_dict_filename,
    ruleset_order,
    predefined_reject=False,
    info_idx=None,
    timestamp_unit="ms",
    sampling_rate=None,
    start_datetime=None,
    split_type=0,
    duration=30,
    overlapping=None,
    peak_detector=6,
    segment_name=None,
    save_image=False,
    output_dir=None,
    delete_signal=False,
):
    """
    Extracts SQIs for PPG, classifies segments, and saves accepted/rejected segments.

    Parameters
    ----------
    All parameters are similar to `get_ppg_sqis` with the addition of:
    rule_dict_filename : str
        Path to the rule dictionary file.
    ruleset_order : dict
        Order of rulesets for classification.
    predefined_reject : bool, optional
        If True, use predefined rejection criteria (default is False).
    save_image : bool, optional
        If True, saves segment images (default is False).
    output_dir : str, optional
        Directory to save accepted/rejected segments (default is current directory).
    delete_signal : bool, optional
        Whether to delete original signals after segmentation (default is True).

    Returns
    -------
    signal_obj
        Signal object containing classified segments and SQIs.
    """
    output_dir = output_dir or os.getcwd()
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."

    # Step 1: Extract SQIs
    segments, signal_obj = get_ppg_sqis(
        file_name,
        signal_idx,
        timestamp_idx,
        sqi_dict_filename,
        info_idx,
        timestamp_unit,
        sampling_rate,
        start_datetime,
        split_type,
        duration,
        overlapping,
        peak_detector,
        delete_signal,
    )

    # Step 2: Load rule dictionary
    try:
        with open(rule_dict_filename, "r") as f:
            rule_dict = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Rule dictionary file not found: {rule_dict_filename}"
        ) from e

    # Step 3: Validate `ruleset_order` against extracted SQIs
    sqi_df = signal_obj.sqis[0]  # Assume single-channel SQI
    missing_sqi_keys = [
        key for key in ruleset_order.values() if key not in sqi_df.columns
    ]
    if missing_sqi_keys:
        raise KeyError(
            f"The following SQIs in `ruleset_order` are missing from the extracted SQIs: {missing_sqi_keys}"
        )

    # Step 4: Classify SQIs
    signal_obj.ruleset, signal_obj.sqis = classify_segments(
        signal_obj.sqis,
        rule_dict_filename,
        ruleset_order,
    )

    # Step 5: Handle predefined reject or generate decisions
    reject_decision = (
        get_reject_segments(segments, wave_type="PPG")
        if predefined_reject
        else ["accept"] * len(signal_obj.sqis[0])
    )
    a_segments, r_segments = get_decision_segments(
        segments, signal_obj.sqis[0]["decision"].to_list(), reject_decision
    )

    # Step 6: Save accepted and rejected segments
    for seg_type, segments_to_save in [("accept", a_segments), ("reject", r_segments)]:
        seg_dir = os.path.join(output_dir, seg_type)
        img_dir = os.path.join(seg_dir, "img") if save_image else None
        os.makedirs(seg_dir, exist_ok=True)
        if save_image:
            os.makedirs(img_dir, exist_ok=True)
        save_segment(
            segments_to_save,
            segment_name=segment_name,
            save_file_folder=seg_dir,
            save_image=save_image,
            save_img_folder=img_dir,
        )

    return signal_obj


def get_ecg_sqis(
    file_name,
    sqi_dict_filename,
    file_type,
    signal_idx=1,
    timestamp_idx=0,
    # channel_num=None,
    # channel_name=None,
    sampling_rate=None,
    start_datetime=None,
    split_type=0,
    duration=30,
    overlapping=None,
    peak_detector=6,
):
    """
    Computes SQIs for ECG segments and returns the segments along with the SQIs.

    Parameters
    ----------
    All parameters are similar to `get_ppg_sqis` with the addition of:
    file_type : str
        Type of the ECG file.
    channel_num : int, optional
        Number of channels in the ECG signal (default is None).
    channel_name : list, optional
        Names of channels in the ECG signal (default is None).

    Returns
    -------
    tuple
        Segments list and signal object with SQIs.
    """
    signal_obj = ECG_reader(
        file_name=file_name,
        file_type=file_type,
        # channel_num=channel_num,
        # channel_name=channel_name,
        sampling_rate=sampling_rate,
        start_datetime=start_datetime,
    )

    segments_lst, milestones_lst = [], []
    # for i in range(1, len(signal_obj.signals.columns)):
    signals = signal_obj.signals.iloc[:, [signal_idx]]
    segments, milestones = split_segment(
        signals,
        split_type=split_type,
        sampling_rate=signal_obj.sampling_rate,
        duration=duration,
        overlapping=overlapping,
        peak_detector=peak_detector,
        wave_type="ECG",
    )
    segments_lst.append(segments)
    milestones_lst.append(milestones)

    signal_obj.signals = pd.DataFrame()
    signal_obj.sqis = [
        extract_sqi(segments, milestones, sqi_dict_filename, wave_type="ECG")
        for segments, milestones in zip(segments_lst, milestones_lst)
    ]
    return segments_lst, signal_obj


def get_qualified_ecg(
    file_name,
    file_type,
    sqi_dict_filename,
    rule_dict_filename,
    ruleset_order,
    signal_idx=1,
    timestamp_idx=0,
    channel_num=None,
    channel_name=None,
    predefined_reject=False,
    sampling_rate=None,
    start_datetime=None,
    split_type=0,
    duration=30,
    auto_mode=False,
    lower_bound=0.1,
    upper_bound=0.9,
    overlapping=None,
    peak_detector=6,
    segment_name=None,
    save_image=False,
    output_dir=None,
):
    """
    Extracts SQIs for ECG, classifies segments, and saves accepted/rejected segments.

    Parameters
    ----------
    All parameters are similar to `get_qualified_ppg` with the addition of:
    file_type : str
        Type of the ECG file.

    Returns
    -------
    signal_obj
        Signal object containing classified segments and SQIs.
    """
    output_dir = output_dir or os.getcwd()
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."

    segment_lst, signal_obj = get_ecg_sqis(
        file_name,
        sqi_dict_filename,
        file_type,
        signal_idx,
        timestamp_idx,
        # channel_num,
        # channel_name,
        sampling_rate,
        start_datetime,
        split_type,
        duration,
        overlapping,
        peak_detector,
    )

    for i, segments in enumerate(segment_lst):
        # Classify and get reject decisions
        signal_obj.ruleset, sqis = classify_segments(
            signal_obj.sqis,
            rule_dict_filename,
            ruleset_order,
            auto_mode,
            lower_bound,
            upper_bound,
        )
        reject_decision = (
            get_reject_segments(segments, wave_type="ECG")
            if predefined_reject
            else ["accept"] * len(sqis[i])
        )

        # Separate accepted and rejected segments
        a_segments, r_segments = get_decision_segments(
            segments, sqis[i]["decision"], reject_decision
        )

        # Save segments
        for seg_type, segments_to_save in [
            ("accept", a_segments),
            ("reject", r_segments),
        ]:
            seg_dir = os.path.join(output_dir, str(i), seg_type)
            img_dir = os.path.join(seg_dir, "img") if save_image else None
            os.makedirs(seg_dir, exist_ok=True)
            if save_image:
                os.makedirs(img_dir, exist_ok=True)
            save_segment(
                segments_to_save,
                segment_name=segment_name,
                save_file_folder=seg_dir,
                save_image=save_image,
                save_img_folder=img_dir,
            )

    return signal_obj


# import tempfile

# if __name__ == "__main__":
#     # Example-based input files and parameters
#     # file_in = os.path.abspath("tests/test_data/ppg_smartcare.csv")
#     file_in = os.path.abspath("tests/test_data/example.edf")
#     sqi_dict = os.path.abspath("tests/test_data/sqi_dict.json")
#     rule_dict_filename = os.path.abspath("tests/test_data/rule_dict_test.json")
#     ruleset_order = {2: "skewness_1", 1: "perfusion"}
#     # output_dir = tempfile.gettempdir()
#     output_dir = "D:\Workspace\Oucru\\vital_sqi\outdir"

#     # Call the function under test
#     signal_obj = get_qualified_ecg(
#         file_name=file_in,
#         sqi_dict_filename=sqi_dict,
#         # signal_idx=6,
#         # timestamp_idx=0,
#         file_type="edf",  # File type explicitly defined as in the example
#         duration=30,  # Duration parameter passed
#         rule_dict_filename=rule_dict_filename,
#         ruleset_order=ruleset_order,
#         output_dir=output_dir,
#     )

#     # signal_obj = get_qualified_ppg(
#     #         file_name=file_in,
#     #         sqi_dict_filename=sqi_dict,
#     #         signal_idx=6,
#     #         timestamp_idx=0,
#     #         # file_type="edf",  # File type explicitly defined as in the example
#     #         duration=30,      # Duration parameter passed
#     #         rule_dict_filename=rule_dict_filename,
#     #         ruleset_order=ruleset_order,
#     #         output_dir=output_dir,
#     # )

#     print(signal_obj.signals[0:100])
