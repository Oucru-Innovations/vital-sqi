import numpy
import numpy as np
import pandas as pd
import vital_sqi.preprocess.preprocess_signal as sqi_pre
import heartpy as hp
import os
import  json
from tqdm import tqdm
from hrvanalysis import get_nn_intervals
from vital_sqi.common.band_filter import BandpassFilter
from heartpy.analysis import calc_ts_measures, calc_rr, calc_fd_measures,\
    clean_rr_intervals, calc_poincare, calc_breathing
from heartpy.peakdetection import check_peaks
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.common.utils import create_rule_def
from vital_sqi.rule import RuleSet, Rule, update_rule
import warnings
import inspect
import vital_sqi.sqi as sq

sqi_nn_list = ["nn_mean_sqi",
               "sdnn_sqi", "sdsd_sqi",
               "rmssd_sqi", "cvsd_sqi",
               "cvnn_sqi", "pnn_sqi",
               "mean_nn_sqi", "median_nn_sqi",
               "hr_mean_sqi", "hr_median_sqi"
                              "hr_min_sqi", "hr_max_sqi", "hr_range_sqi",
               "peak_frequency_sqi", "absolute_power_sqi",
               "log_power_sqi", "relative_power_sqi",
               "normalized_power_sqi", "lf_hf_ratio_sqi",
               "poincare_sqi", "hrva"
               ]

def classify_segments(sqis, rule_dict, ruleset_order):
    ruleset = {}
    for i in ruleset_order:
        rule = Rule(ruleset_order[i])
    rule.load_def(rule_dict)
    ruleset[i] = rule
    sqis['decision'] = ruleset.execute(sqis)
    return ruleset, sqis


def get_decision_segments(segments, decision):
    a_segments,r_segments = None
    return a_segments, r_segments


def extract_sqi(segments, milestones, sqi_dict):
    # return sqis pandas Dataframe with milestones info
    sqis = None
    return sqis


def get_all_features_heartpy(data_sample, sample_rate=100, rpeak_detector=0):
    """

    Parameters
    ----------
    data_sample :
        Raw signal

    sample_rate :
        (Default value = 100)
    rpeak_detector :
        (Default value = 0)

    Returns
    -------


    """
    # time domain features
    td_features = ["bpm", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50",
                   "hr_mad", "sd1", "sd2", "s", "sd1/sd2", "breathingrate"]
    # frequency domain features
    fd_features = ["lf", "hf", "lf/hf"]
    try:
        wd, m = hp.process(data_sample, sample_rate, calc_freq=True)
    except Exception as e:
        try:
            wd, m = hp.process(data_sample, sample_rate)
        except:
            time_domain_features = {k: np.nan for k in td_features}
            frequency_domain_features = {k: np.nan for k in fd_features}
            return time_domain_features, frequency_domain_features
    if rpeak_detector in [1, 2, 3, 4]:
        detector = PeakDetector(wave_type='ecg')
        peak_list = \
        detector.ppg_detector(data_sample, rpeak_detector, preprocess=False)[0]
        wd["peaklist"] = peak_list
        wd = calc_rr(peak_list, sample_rate, working_data=wd)
        wd = check_peaks(wd['RR_list'], wd['peaklist'], wd['ybeat'],
                         reject_segmentwise=False, working_data=wd)
        wd = clean_rr_intervals(working_data=wd)
        rr_diff = wd['RR_list']
        rr_sqdiff = np.power(rr_diff, 2)
        wd, m = calc_ts_measures(wd['RR_list'], rr_diff, rr_sqdiff,
                                 working_data=wd)
        m = calc_poincare(wd['RR_list'], wd['RR_masklist'], measures=m,
                          working_data=wd)
        try:
            measures, working_data = calc_breathing(wd['RR_list_cor'],
                                                    data_sample, sample_rate,
                                                    measures=m, working_data=wd)
        except:
            measures['breathingrate'] = np.nan

        wd, m = calc_fd_measures(measures=measures, working_data=working_data)

    time_domain_features = {k: m[k] for k in td_features}

    frequency_domain_features = {}
    for k in fd_features:
        if k in m.keys():
            frequency_domain_features[k] = m[k]
        else:
            frequency_domain_features[k] = np.nan
    # frequency_domain_features = {k:m[k] for k in fd_features if k in m.keys}
    # frequency_domain_features = {k:np.na for k in fd_features if k not in m.keys}

    return time_domain_features, frequency_domain_features


# #Expecting filtered signal
# def calculate_SQI(waveform_segment, trough_list, taper, sqi_dict):
#     """
#     Calculate SQI function to calculate each SQI that is part of the VITAL SQI package.
#     The functions that should be calculated are supplied in an input dictonary, including optional
#     parameters. The main wrapper function around individual SQI functions.
#
#     Parameters
#     ----------
#     waveform_segment : array or dataframe
#         Segment of PPG or ECG waveform on which we will perform SQI extraction
#
#     trough_list : array of int
#         Idices of troughs in the signal provided by peak detector to be able to extract individual beats
#
#     taper : bool
#         Enable tapering for per beat SQI extraction
#
#     sqi_dict : dict
#         SQI dictonary where keys match with abbreviations of the SQI functions. Each key contains a 2 element touple.
#         The first elemnt is a dictonary with additional parameters for SQI funciton, second element is string that
#         describes if the given SQI function should be executed per beat, per segment or per both.
#
#     Returns
#     -------
#     calculated_SQI : dict
#         A dictonary of all computed SQI values
#
#     """
#     variations_stats = ['_list','_mean','_median','_std']
#     SQI_dict = {}
#     for sqi_func, args in sqi_dict.items():
#     #sqi_func is the function handle extracted from dictonary
#     # args[0] = function arguments to calculate SQI
#     # args[1] = per segment, per beat or both
#         if 'beat' in args[1]:
#             #Do per beat calculation
#             if args[0] != {}:
#                 SQI_list = per_beat_sqi(MASTER_SQI_DICT[sqi_func], trough_list, waveform_segment, taper, **args[0])
#             else:
#                 SQI_list = per_beat_sqi(MASTER_SQI_DICT[sqi_func], trough_list, waveform_segment, taper)
#             SQI_dict[sqi_func+variations_stats[0]] = SQI_list
#             SQI_dict[sqi_func+variations_stats[1]] = np.mean(SQI_list)
#             SQI_dict[sqi_func+variations_stats[2]] = np.median(SQI_list)
#             SQI_dict[sqi_func+variations_stats[3]] = np.std(SQI_list)
#         if 'segment' in args[1]:
#             #Do per segment calculation
#             if args[0] != {}:
#                 SQI_dict[sqi_func] = MASTER_SQI_DICT[sqi_func](waveform_segment, **args[0])
#             else:
#                 SQI_dict[sqi_func] = MASTER_SQI_DICT[sqi_func](waveform_segment)
#
#     return pd.Series(SQI_dict)


def per_beat_sqi(sqi_func, troughs, signal, taper=False, **kwargs):
    """
    Perform a per-beat application of the selected SQI function on the signal
    segment

    Parameters
    ----------
    sqi_func : function
        An SQI function to be performed.

    troughs : array of int
        Idices of troughs in the signal provided by peak detector to be able to extract individual beats

    signal :
        Signal array containing one segment of the waveform

    taper : bool
        Is each beat need to be tapered or not before executing the SQI function

    **kwargs : dict
        Additional positional arguments that needs to be fed into the SQI function

    Returns
    -------
    calculated_SQI : array
        An array with SQI values for each beat of the signal

    """
    #Remove first and last trough as they might be on the edge
    troughs = troughs[1:-1]
    if len(troughs) > 2:
        sqi_vals = []
        for idx, beat_start in enumerate(troughs[:-1]):
            single_beat = signal[beat_start:troughs[idx+1]]
            if taper:
                single_beat = sqi_pre.taper_signal(single_beat)
            if len(kwargs) != 0:
                args = tuple(kwargs.values())
                sqi_vals.append(sqi_func(single_beat, *args))
            else:
                sqi_vals.append(sqi_func(single_beat))
        return sqi_vals

    else:
        return [-np.inf]
        raise Exception("Not enough peaks in the signal to generate per beat SQI")


def get_sqi_dict(sqis, sqi_name):
    """

    :param sqis:
    :param sqi_name:
    :return:
    """
    if sqi_name == 'correlogram_sqi':
        SQI_dict = {}
        variations_acf = ['_peak1', '_peak2', '_peak3', '_value1', '_value2', '_value3']
        for idx, variations in enumerate(variations_acf):
            SQI_dict['correlogram' + variations+"_sqi"] = sqis[idx]
        return SQI_dict

    if isinstance(sqis, dict):
        return sqis

    if isinstance(sqis, (float, int)):
        return {sqi_name: sqis}

    if isinstance(sqis,numpy.ndarray):
        if len(sqis.shape) == 0:
            return {sqi_name: -1}
        return {sqi_name: sqis[0]}

    if isinstance(sqis, list):
        SQI_dict = {}
        variations_stats = ['_mean', '_median', '_std']
        SQI_dict[sqi_name.split("_")[0] + variations_stats[0]+"_sqi"] = np.mean(sqis)
        SQI_dict[sqi_name.split("_")[0] + variations_stats[1]+"_sqi"] = np.median(sqis)
        SQI_dict[sqi_name.split("_")[0] + variations_stats[2]+"_sqi"] = np.std(sqis)
        return SQI_dict

    if isinstance(sqis, tuple):
        SQI_dict = {}
        for features_dict in sqis:
            features_dict_ = dict((key+"_sqi", value) for (key, value) in features_dict.items())
            SQI_dict = {**SQI_dict, **features_dict_}
        return SQI_dict


def get_sqi(sqi_func, s, per_beat=False,
            wave_type='ppg',peak_detector=7,
            **kwargs):
    signal_arg = inspect.getfullargspec(sqi_func)[0][0]
    if signal_arg == 'nn_intervals':
        s = get_nn(s.iloc[:, 1])
    else:
        s = s.iloc[:,1]
    if per_beat:
        # Prepare primary peak detector and perform peak detection
        detector = PeakDetector()
        if wave_type =='ppg':
            peak_list, trough_list = detector.ppg_detector(s,
                                                    peak_detector)
        else:
            peak_list, trough_list = detector.ecg_detector(s,
                                                    peak_detector)
        sqi_scores = per_beat_sqi(sqi_func, trough_list, s, **kwargs)
    else:
        if 'wave_type' in inspect.getfullargspec(sqi_func)[0]:
            kwargs['wave_type'] = wave_type
        sqi_scores = sqi_func(s,**kwargs)
    sqi_name = sqi_func.__name__
    sqi_score_dict = get_sqi_dict(sqi_scores,sqi_name)
    return sqi_score_dict


def segment_PPG_SQI_extraction(sig,sqi_list,sqi_arg_list):
    """

    :param sig:
    :param sqi_list: list of sqi as in MASTERDICT
    :param nn_sqi_list: list of sqi using nn_intervals as in 'HRV' MASTER_DICT
    :param nn_sqi_arg_list:
    :param sqi_arg_list:
    :return:
    """
    sqi_score = {}
    for (sqi_,args_) in zip(sqi_list,sqi_arg_list):
        try:
            sqi_score = {**sqi_score, **get_sqi(sqi_, sig, **args_)}
        except Exception as err:
            print(sqi_)
            print(err)
            continue
    return pd.Series(sqi_score)


def compute_SQI(signal, segment_length='30s', primary_peakdet=7, secondary_peakdet=6, wave_type='ppg', sampling_rate=100, template_type=1):
    if wave_type == 'ppg':
        try:
            sqis = signal.groupby(pd.Grouper(freq=segment_length)).apply(segment_PPG_SQI_extraction)
        except Exception as e:
            return None
    # elif wave_type == 'ecg':
    #     sqis = signal.groupby(pd.Grouper(freq=segment_length)).apply(segment_ECG_SQI_extraction, sampling_rate, primary_peakdet, secondary_peakdet, (1, 1), (20, 4), template_type)
    else:
        raise Exception("Wrong type of waveform supplied. Only accepts 'ppg' or 'ecg'.")
    return sqis



def extract_sqi(segments,milestones,sqi_list,file_name,arg_path=None):
    if arg_path == None:
        arg_path = os.path.join(os.getcwd(),"../resource/sqi_args.json")
    with open(arg_path, 'r') as arg_file:
        sqi_arg_list = json.loads(arg_file.read())
    # nn_sqi_arg_list = sqi_list
    df_sqi = pd.DataFrame()
    for segment_idx in tqdm(range(len(segments))):
        segment = segments[segment_idx]
        sqi_arg_list['perf'] = {'y':segment.iloc[:,1]}
        # sqi_arg_list['zc'] = {'y': ppg_stable.iloc[:, 1]}
        # sqi_arg_list['mc'] = {'y': ppg_stable.iloc[:, 1]}
        sqis = segment_PPG_SQI_extraction(segment, sqi_list.values(),sqi_arg_list.values())

        # segment_name_list = [file_name.split("/")[-1] + "_" + str(i) for i in range(len(sqis))]
        segment_name_list = file_name.split("/")[-1] + "_" +str(segment_idx)
        sqis['id'] = segment_name_list
        df_sqi = df_sqi.append(sqis, ignore_index=True)
    return df_sqi

# Trong ruleset class??
def generate_rule(rule_name, rule_def):
    rule_def, boundaries, label_list = update_rule(rule_def, is_update=False)
    rule_detail = {'def': rule_def,
                     'boundaries': boundaries,
                     'labels': label_list}
    rule = Rule(rule_name,rule_detail)
    return rule


def get_decision(df_sqi, selected_rule, json_rule_dict):
    rule_list = {}
    for (i, selected_sqi) in zip(range(len(selected_rule)), selected_rule):
        rule = generate_rule(selected_sqi, json_rule_dict[selected_sqi]['def'])

        rule_list[i + 1] = rule
    ruleset = RuleSet(rule_list)

    decision_list = []
    for idx in range(len(df_sqi)):
        row_data = pd.DataFrame(dict(df_sqi[selected_rule].iloc[idx]), index=[0])
        decision_list.append(ruleset.execute(row_data))

    return decision_list


def get_nn(s,wave_type='ppg',sample_rate=100,rpeak_method=7,remove_ectopic_beat=False):
    if wave_type=='ppg':
        detector = PeakDetector(wave_type='ppg')
        peak_list, trough_list = detector.ppg_detector(s, detector_type=rpeak_method)
    else:
        detector = PeakDetector(wave_type='ecg')
        peak_list, trough_list = detector.ecg_detector(s, detector_type=rpeak_method)

    rr_list = np.diff(peak_list) * (1000 / sample_rate)
    if not remove_ectopic_beat:
        return rr_list
    nn_list = get_nn_intervals(rr_list)
    nn_list_non_na = np.copy(nn_list)
    nn_list_non_na[np.where(np.isnan(nn_list_non_na))[0]] = -1
    return nn_list_non_na



def example_rule_decision(df_sqi):
    selected_rule = ['msq', 'entropy_std']
    for single_rule in selected_rule:
        boundary = np.around(np.quantile(df_sqi[single_rule], [0.05, 0.95]), decimals=2)
        upper_bound = boundary[1]
        lower_bound = boundary[0]
        json_rule_dict = create_rule_def(single_rule,upper_bound, lower_bound)
    decision_list = get_decision(df_sqi,selected_rule,json_rule_dict)
    df_sqi['decision'] = decision_list
