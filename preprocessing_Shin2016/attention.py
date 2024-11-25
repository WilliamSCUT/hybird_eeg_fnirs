import scipy
import numpy as np
from AttentionAlgorithms import AttentionAlgorithms


def compute_attention(raw_data_, original_sampling_rate, resampling_rate, interval, algorithm):
    """
    compute attention

    """
    raw_data = raw_data_  # in shape of (15, 600)
    # raw_data = raw_data[:, :, 1] * 1e6  # convert to microvolts

    channels = raw_data.shape[0]
    data_len = raw_data.shape[1]
    new_frequency = resampling_rate  # ##
    original_frequency = original_sampling_rate
    resampling_factor = new_frequency / original_frequency

    # spectral_features_list = ['spectral_power']
    spectral_features_list = ['spectral_power_by_scipy']

    res = []
    att = []

    new_ = True  # 1-(0+2)/1

    for c in range(channels):
        # filter data
        # raw_data[c] = filter_with_PMD(data=raw_data[c], Fs = original_sampling_rate, wn=6, lp=60)

        # resampling data
        raw_data_resampled = scipy.signal.resample_poly(raw_data[c], up=new_frequency, down=original_frequency) if resampling_factor != 1 else raw_data[c]
        # resampled_data = resampled_data - np.mean(resampled_data)  # remove DC offset

        for feature_name in spectral_features_list:  # ['spectral_power_by_scipy']
            # compute sum of spectral power with respect to frequency bands
            # delta(0.5-4), theta(4-7), alpha(7-13), beta(13-30)
            # 0             1           2           3
            _ = spectral_features(raw_data_resampled, new_frequency, feature_name)
            res.append([.0, .0, .0, .0] if np.isnan(_).any() else _)
        if np.sum(res[-1]) != 0:

            if algorithm == AttentionAlgorithms.Ec:
                att.append(AttentionAlgorithms.compute_ec(res[c]))  # R=Beat/(Alpha +Theta)
            elif algorithm == AttentionAlgorithms.Proportion_of_Beta:
                att.append(AttentionAlgorithms.compute_proportion_of_beta(res[c]))  # R=Beat/(Alpha +Total)
            elif algorithm == AttentionAlgorithms.Beta_over_Alpha:
                att.append(AttentionAlgorithms.compute_beta_over_alpha(res[c]))
            elif algorithm == AttentionAlgorithms.XY_RATIO:
                att.append(AttentionAlgorithms.compute_xy_ratio(res[c]))
            else: # default
                att.append(0)
        else:
            att.append(0)
    if new_:  # att[1]-(att[0]+att[2])/2
        # return (
        #     att[0], np.max(att), np.min(att),
        #     [np.mean(np.stack(res)[:, 0]), np.mean(np.stack(res)[:, 1]), np.mean(np.stack(res)[:, 2]),
        #      np.mean(np.stack(res)[:, 3])]
        # )
        att = [_ for _ in att if _ != 0]
        res = [_ for _ in res if np.sum(_) != 0]
        return (
            np.mean(att),np.array(res)
        )
    return (
        np.mean(att), np.max(att), np.min(att),
        [np.mean(np.stack(res)[:, 0]), np.mean(np.stack(res)[:, 1]), np.mean(np.stack(res)[:, 2]),
         np.mean(np.stack(res)[:, 3])]
    )
# 计算一个channel的PSD
def spectral_features(x, new_frequency, feature_name='spectral_power', params_st=[]):  # Fs = 1000, feature_name='spectral_power'
    if len(params_st) == 0:
        if 'spectral' in feature_name:
            params_st = SpectralParameters(method = 'periodogram')
            # params_st = SpectralParameters(method='welch_periodogram')
    freq_bands = params_st.freq_bands

    if len(x) < (params_st.L_window * new_frequency):
        print('SPECTRAL features: signal length < window length; set shorter L_window')
        featx = np.nan
        return featx
    if feature_name == 'spectral_power':
        # ---------------------------------------------------------------------
        # apply spectral analysis with periodogram or welch periodogram
        # ---------------------------------------------------------------------
        # TODO. implement periodogram and welch periodogram
        pass
    elif feature_name == 'spectral_power_by_scipy':

        # nperseg 窗口点数
        # noverlap 重叠点数
        frequencies, psd = scipy.signal.welch(x, new_frequency, nperseg=new_frequency*2, noverlap=new_frequency)
        delta_power = calculate_band_power(frequencies, psd, params_st.freq_bands[0])
        theta_power = calculate_band_power(frequencies, psd, params_st.freq_bands[1])
        alpha_power = calculate_band_power(frequencies, psd, params_st.freq_bands[2])
        beta_power = calculate_band_power(frequencies, psd, params_st.freq_bands[3])
        return np.array([delta_power, theta_power, alpha_power, beta_power])


def calculate_band_power(f, Pxx_den, band):
    indices = np.logical_and(f >= band[0], f <= band[1])
    band_power = np.trapz(Pxx_den[indices], f[indices])  # integral
    return band_power

class SpectralParameters:
    def __init__(self, method='PSD', L_window=2, window_type='hamm', overlap=50,
                 freq_bands=np.array([[0.5, 4], [4, 7], [7, 13], [13, 30]]),
                 total_freq_bands=None, SEF=0.95):
        # 初始化与频谱分析相关的参数
        # how to estimate the spectrum for 'spectral_flatness', 'spectral_entropy',
        # spectral_edge_frequency features:
        # 1) PSD: estimate power spectral density (e.g. Welch periodogram)
        # 2) robust-PSD: median (instead of mean) of spectrogram
        # 3) periodogram: magnitude of the discrete Fourier transform

        self.method = method

        # length of time - domain analysis window and overlap:
        # (applies to 'spectral_power', 'spectral_relative_power',
        # 'spectral_flatness', and 'spectral_diff' features)
        self.L_window = L_window  # in seconds
        self.window_type = window_type  # type of window
        self.overlap = overlap  # overlap in percentage
        self.freq_bands = freq_bands
        if total_freq_bands is None:
            total_freq_bands = [0.5, 30]
        self.total_freq_bands = total_freq_bands
        self.SEF = SEF  # spectral edge frequency


def normalize(value, min_old=0.0, max_old=2.0, min_new=0, max_new=100):
    value_clipped = np.clip(value, min_old, max_old)
    normalized_value = (value_clipped - min_old) / (max_old - min_old) * (max_new - min_new) + min_new
    return normalized_value