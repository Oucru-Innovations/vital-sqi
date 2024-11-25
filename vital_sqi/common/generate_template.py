"""Generating templates of ECG and PPG complexes"""

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from scipy import signal, interpolate
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp


def squeeze_template(s, width):
    """
    Compress a signal template by averaging values within a given width.

    Parameters
    ----------
    s : array_like
        Input signal array.
    width : int
        Desired compressed length of the output signal.

    Returns
    -------
    numpy.ndarray
        Compressed signal.
    """
    s = np.array(s)
    total_len = len(s)
    span_unit = 2
    out_res = []
    for i in range(int(width)):
        centroid = (total_len / width) * i
        left_point = max(0, int(centroid) - span_unit)
        right_point = min(len(s), int(centroid + span_unit))
        out_res.append(np.mean(s[left_point:right_point]))
    return np.array(out_res)


def ppg_dual_double_frequency_template(width):
    """
    Generate a PPG template using two sine waveforms with different frequencies.

    Parameters
    ----------
    width : int
        The sample size of the generated waveform.

    Returns
    -------
    numpy.ndarray
        A 1-D array representing the PPG waveform with a diastolic peak at a low position.
    """
    t = np.linspace(0, 1, width, False)  # 1 second
    sig = np.sin(2 * np.pi * 2 * t - np.pi / 2) + np.sin(2 * np.pi * 1 * t - np.pi / 6)
    sig_scale = MinMaxScaler().fit_transform(sig.reshape(-1, 1))
    return sig_scale.reshape(-1)


def skew_func(x, e=0, w=1, a=0):
    """
    Generate a skewness distribution.

    Parameters
    ----------
    x : array_like
        Input sequence of time points.
    e : float, optional
        Location parameter (default is 0).
    w : float, optional
        Scale parameter (default is 1).
    a : float, optional
        Shape parameter (default is 0).

    Returns
    -------
    numpy.ndarray
        A 1-D array of a skewness distribution.
    """
    t = (x - e) / w
    omega = (1 + erf((a * t) / np.sqrt(2))) / 2
    gaussian_dist = 1 / (np.sqrt(2 * np.pi)) * np.exp(-(t**2) / 2)
    return 2 / w * gaussian_dist * omega


def ppg_absolute_dual_skewness_template(width, e_1=1, w_1=2.5, e_2=3, w_2=3, a=4):
    """
    Generate a PPG template using two skewness distributions.

    Parameters
    ----------
    width : int
        Sample size of the generated waveform.
    e_1, w_1 : float, optional
        Parameters for the first skew distribution (default e_1=1, w_1=2.5).
    e_2, w_2 : float, optional
        Parameters for the second skew distribution (default e_2=3, w_2=3).
    a : float, optional
        Shape parameter for both distributions (default is 4).

    Returns
    -------
    numpy.ndarray
        A 1-D array representing the PPG waveform.
    """
    x = np.linspace(0, 11, width, False)
    p_1 = skew_func(x, e_1, w_1, a)
    p_2 = skew_func(x, e_2, w_2, a)
    p_ = np.max([p_1, p_2], axis=0)
    sig_scale = MinMaxScaler().fit_transform(p_.reshape(-1, 1))
    return sig_scale.reshape(-1)


def ppg_nonlinear_dynamic_system_template(width):
    """
    Generate a PPG template based on a nonlinear dynamic system.

    Parameters
    ----------
    width : int
        Desired length of the template.

    Returns
    -------
    numpy.ndarray
        A rescaled signal template for PPG.
    """
    x1, x2, u = 0.15, 0.15, 0.5
    beta, gamma1, gamma2 = 1, -0.25, 0.25
    x1_list, x2_list = [x1], [x2]
    dt = 0.1

    for _ in np.arange(1, 100, dt):
        y1 = 0.5 * (np.abs(x1 + 1) - np.abs(x1 - 1))
        y2 = 0.5 * (np.abs(x2 + 1) - np.abs(x2 - 1))
        dx1 = -x1 + (1 + u) * y1 - beta * y2 + gamma1
        dx2 = -x2 + (1 + u) * y2 + beta * y1 + gamma2

        x1 += dx1 * dt
        x2 += dx2 * dt

        x1_list.append(x1)
        x2_list.append(x2)

    local_minima = argrelextrema(np.array(x2_list), np.less)[0]
    s = np.array(x2_list[local_minima[-2] : local_minima[-1] + 1])
    rescale_signal = squeeze_template(s, width)

    window = signal.windows.cosine(len(rescale_signal), 0.5)
    signal_data_tapered = window * (rescale_signal - min(rescale_signal))

    out_scale = MinMaxScaler().fit_transform(signal_data_tapered.reshape(-1, 1))
    return out_scale.reshape(-1)


def interp(ys, mul):
    """
    Perform cubic interpolation and extrapolation on a sequence.

    Parameters
    ----------
    ys : array_like
        Input sequence.
    mul : int
        Multiplication factor for interpolation.

    Returns
    -------
    numpy.ndarray
        Interpolated sequence.
    """
    ys = list(ys)
    ys.append(2 * ys[-1] - ys[-2])
    xs = np.arange(len(ys))
    fn = interpolate.interp1d(xs, ys, kind="cubic")
    new_xs = np.arange(len(ys) - 1, step=1.0 / mul)
    return fn(new_xs)


def ecg_dynamic_template(
    width,
    sfecg=256,
    N=256,
    Anoise=0,
    hrmean=60,
    hrstd=1,
    lfhfratio=0.5,
    sfint=512,
    ti=np.array([-70, -15, 0, 15, 100]),
    ai=np.array([1.2, -5, 30, -7.5, 0.75]),
    bi=np.array([0.25, 0.1, 0.1, 0.1, 0.4]),
):
    """
    Generate a synthetic ECG signal template.

    Parameters
    ----------
    width : int
        Desired length of the output template.
    sfecg, N, Anoise, hrmean, hrstd, lfhfratio, sfint : int or float, optional
        Parameters for the ECG generation model.
    ti, ai, bi : numpy.ndarray, optional
        Arrays of model parameters.

    Returns
    -------
    numpy.ndarray
        Generated ECG template.
    """
    ti = ti * np.pi / 180
    hrfact = np.sqrt(hrmean / 60)
    hrfact2 = np.sqrt(hrfact)
    bi = hrfact * bi
    ti = np.multiply([hrfact2, hrfact, 1, hrfact, hrfact2], ti)

    flo, fhi, flostd, fhistd = 0.1, 0.25, 0.01, 0.01
    rr0 = rr_process(
        flo,
        fhi,
        flostd,
        fhistd,
        lfhfratio,
        hrmean,
        hrstd,
        1,
        2 ** int(np.ceil(np.log2(N * 60 / hrmean))),
    )
    rr = interp(rr0, sfint)
    dt = 1 / sfint
    rrn = np.zeros(len(rr))
    tecg, i, Nt = 0, 0, len(rr)

    while i < Nt:
        tecg += rr[i]
        ip = int(np.round(tecg / dt))
        rrn[i : ip + 1] = rr[i]
        i = ip + 1

    tspan = np.arange(0, (ip - 1) * dt, dt)
    args = (rrn, sfint, ti, ai, bi)
    solv_ode = solve_ivp(
        ordinary_differential_equation,
        [tspan[0], tspan[-1]],
        [1, 0, 0.04],
        t_eval=np.arange(20.5, 21.5, 0.00001),
        args=args,
    )
    return solv_ode.y[2]


def ordinary_differential_equation(t, x_equations, rr, sfint, ti, ai, bi):
    """
    Solve the ordinary differential equation for synthetic ECG generation.

    Parameters
    ----------
    t : float
        Time variable.
    x_equations : array_like
        Initial values for x, y, and z variables.
    rr : numpy.ndarray
        Resampled RR interval sequence.
    sfint : int
        Sampling frequency for interpolation.
    ti, ai, bi : numpy.ndarray
        Model parameters for ECG generation.

    Returns
    -------
    list
        Derivatives of x, y, and z.
    """
    x, y, z = x_equations
    ta = np.arctan2(y, x)
    a0 = 1.0 - np.sqrt(x**2 + y**2)
    ip = int(1 + np.floor(t * sfint))
    w0 = 2 * np.pi / (rr[ip] if ip < len(rr) else rr[-1])

    dx1dt = a0 * x - w0 * y
    dx2dt = a0 * y + w0 * x
    dti = np.fmod(ta - ti, 2 * np.pi)
    dx3dt = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2)) - 1.0 * (
        z - 0.005 * np.sin(2 * np.pi * 0.25 * t)
    )

    return [dx1dt, dx2dt, dx3dt]


def rr_process(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n):
    """
    Generate an RR interval time series with bimodal power spectrum.

    Parameters
    ----------
    flo, fhi : float
        Low and high frequencies of the Gaussian distributions.
    flostd, fhistd : float
        Standard deviations of low and high frequencies.
    lfhfratio : float
        Ratio of low-frequency to high-frequency power.
    hrmean, hrstd : float
        Mean and standard deviation of heart rate.
    sfrr : int
        Sampling frequency for RR intervals.
    n : int
        Length of the generated time series.

    Returns
    -------
    numpy.ndarray
        Generated RR interval time series.
    """
    w1, w2 = 2 * np.pi * flo, 2 * np.pi * fhi
    c1, c2 = 2 * np.pi * flostd, 2 * np.pi * fhistd
    sig2, sig1 = 1, lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)

    df = sfrr / n
    w = np.arange(0, n) * 2 * np.pi * df
    Hw1 = sig1 * np.exp(-0.5 * ((w - w1) / c1) ** 2) / np.sqrt(2 * np.pi * c1**2)
    Hw2 = sig2 * np.exp(-0.5 * ((w - w2) / c2) ** 2) / np.sqrt(2 * np.pi * c2**2)
    Hw = Hw1 + Hw2

    Sw = (sfrr / 2) * np.sqrt(Hw)
    ph = np.concatenate(
        [
            [0],
            2 * np.pi * np.random.rand(int(n / 2) - 1),
            [0],
            -np.flip(2 * np.pi * np.random.rand(int(n / 2) - 1)),
        ]
    )

    SwC = Sw * np.exp(1j * ph)
    x = (1 / n) * np.real(np.fft.ifft(SwC))
    rr = rrmean + (rrstd / np.std(x)) * x
    return rr
