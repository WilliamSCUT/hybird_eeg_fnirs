from tqdm import tqdm
import numpy as np
import itertools
from scipy.io import loadmat
import scipy
import mne


from AttentionAlgorithms import AttentionAlgorithms
from attention import compute_attention

mne.set_log_level('ERROR')


def convert_one_session(data, mark, session_num):
    eeg = data[session_num].x.T * 1e-6  # from uV to V
    trig = np.zeros((1, eeg.shape[1]))  # (1, 1202336)

    # The preprocessed EEG signals were down sampled from 1000Hz to 200 Hz.
    idx = (mark[session_num].time - 1) // 5
    trig_offset = 2


    # trig[0, idx] = mark[session_num].event.desc // 16 + trig_offset
    trig[0, idx] = mark[session_num].event.desc


    eeg = np.vstack([eeg, trig])
    channel_name = list(data[session_num].clab) + ['Stim']
    channel_types = ['eeg'] * 30 + ['eog'] * 2 + ['stim']

    # Create a MNE Raw object
    montage = mne.channels.make_standard_montage('standard_1005')
    # montage.plot(kind='topomap')
    info = mne.create_info(channel_name, 200, channel_types)
    raw = mne.io.RawArray(eeg, info, verbose=False)
    raw.set_montage(montage)

    # apply annotations for each raw object
    # if not, the annotations will be empty, to be revised.
    # TODO

    return {'0': raw}

def extract_epochs(subject_data, config):
    # extract epochs
    fps = config['down_sampling_rate']
    start_of_epoch = int(config['epoch_start'] * fps)
    duration_of_epoch = int((config['epoch_end'] - config['epoch_start']) * fps)

    info = list(subject_data.values())[0]['0'].info

    # 3*20=60 instances in total for each subject
    instance_data = np.empty((60, 33, duration_of_epoch))  # (3 session*20 repetitions) trials, 33 channels, 200 fps * 30s
    instance_events = np.empty((3, 20, 3), dtype=np.int32)  # 3 sessions, 20 repetitions, 3 columns
    instance_durations = np.empty((3,), dtype=np.int32)  # store the length of sessions

    args = list(map(lambda x: [x, start_of_epoch, duration_of_epoch], subject_data.values()))
    results_session = list(itertools.starmap(session_mapper, args))  ############## per session ##############

    for i, result in enumerate(results_session):  # per session
        # data in shape of (20, 33, 6000)
        # event in shape of (20, 3)
        # duration type of int
        data, events, duration = result

        # instance_data in shape of (60, 33, 6000)
        instance_data[i*20:(i+1)*20, :, :] = data
        # instance_events in shape of (3, 20, 3)  每个epoch包含三个信息：开始时间，0，y
        instance_events[i] = events
        # instance_durations in shape of (3,)
        instance_durations[i] = duration

    # instance_events in shape of (3, 20, 3 columns), 3 columns: begin_at, 0, y
    # 给每个session的event加上前一个session的duration，为的是后面的epoch切割 TODO
    instance_events[1, :, 0] += instance_durations[0]
    instance_events[2, :, 0] += np.sum(instance_durations[0:2])
    instance_events = instance_events.reshape((60, 3))

    # construct the EpochsArray
    current_subject_data = mne.EpochsArray(instance_data, info, tmin=config['epoch_start'],
                                           events=instance_events, baseline=config['baseline'])
    current_subject_data.pick(SELECTED_CHANNELS)

    filter_parameters, filter_dict = get_filter_parameters(config)  # TODO. add decorator @lru_cache(maxsize=None)
    current_subject_data.filter(*filter_parameters, **filter_dict)  # TODO. to understand the filter function
    return current_subject_data



def get_filter_parameters(config):
    ma_filter_config = config['filter']['MA'] if config['is_MA'] else None
    assert ma_filter_config is not None, "Error: MA_config is None"

    pass_band_edge = ma_filter_config['Wp']  # (0.0400, 0.3500)
    stop_band_edge = ma_filter_config['Ws']  # (0.0100, 0.3800)
    pass_band_ripple = ma_filter_config['Rp']  # 3
    stop_band_ripple = ma_filter_config['Rs']  # 30
    fps = config['down_sampling_rate']

    # 最小的滤波器阶数 _ord
    _ord, Wn = scipy.signal.cheb2ord(pass_band_edge, stop_band_edge, pass_band_ripple, stop_band_ripple)
    sos = scipy.signal.cheby2(_ord, stop_band_ripple, stop_band_edge, btype='bandpass', output='sos')

    nyquist_f = fps / 2.

    pass_band = [pass_band_edge[0] * nyquist_f, pass_band_edge[1] * nyquist_f]
    stop_band = [stop_band_edge[0] * nyquist_f, stop_band_edge[1] * nyquist_f]

    iir_params = mne.filter.construct_iir_filter({
        'sos': sos,
        'output': 'ba'}, pass_band, stop_band, fps, btype='bandpass')

    return pass_band, {'method': 'iir', 'iir_params': iir_params}


def session_mapper(data, start_at_frame, duration_of_frame):
    # 针对每个session，提取出20个trial的数据和event
    raw = data['0']  # a single session data of a subject
    raw.set_eeg_reference(ref_channels='average')

    times = raw._data[-1]  # stimulator channel, 4 for subtraction, 3 for rest, 0 for nothing
    cs = np.where(times > 0)[0]  # indices of the stimulator time points

    session_data_by_trials = np.empty((20, 33, duration_of_frame))  # 20 repetitions, 33 channels, 200 fps * 30s
    session_events = np.empty((20, 3), dtype=np.int32)  # 20 repetitions, 3 columns
    session_total_duration = raw._data.shape[1]

    for i, x in enumerate(cs):  ############## per trial ##############

        y = int(times[x])  # type of the event
        begin_at = x + start_at_frame
        end_at = begin_at + duration_of_frame
        single_trial = raw._data[:, begin_at: end_at]  # (all channels, 30s * 200fps)

        session_data_by_trials[i] = single_trial
        session_events[i] = [begin_at, 0, y]  # ??? TODO

    #   session_data_by_trials in shape of (20, 33, 6000)
    #   session_events in shape of (20, 3)
    return session_data_by_trials, session_events, session_total_duration


def compute_(current_window_data):
    # current_window_data in shape of (60, 15, 600)
    data = current_window_data.get_data() * 1e6  # from V to uV, in shape of (60, 15, 600)
    labels = current_window_data.events[:, -1]  # 4(32) for rest, 3(16) for subtraction

    trial_att = []
    for trial in range(data.shape[0]):  ############# per trial ##############
        # for each trial
        # get the data of the trial
        trial_data = data[trial]  # in shape of (15, 600)
        # get the label of the trial
        trial_label = labels[trial]  # 4(32) for rest, 3(16) for subtraction

        att_score, bands_integral_c_x_bands_per_second = compute_attention(
            trial_data, 200, 64, 10, AttentionAlgorithms.Ec
        )
        trial_att.append(att_score)
    return np.array(trial_att)
# def spectral_features(channel_data, new_frequency, feature_name):
#
#     # spectral power
#     f, Pxx = scipy.signal.welch(channel_data, fs=new_frequency, nperseg=new_frequency*2, noverlap=new_frequency//2)
#
#     # band power
#     delta = np.mean(Pxx[(f >= 0.5) & (f < 4)])
#     theta = np.mean(Pxx[(f >= 4) & (f < 8)])
#     alpha = np.mean(Pxx[(f >= 8) & (f < 13)])
#     beta = np.mean(Pxx[(f >= 13) & (f < 30)])
#     gamma = np.mean(Pxx[(f >= 30) & (f < 50)])
#
#     # # attention
#     # attention = (beta + gamma) / (delta + theta + alpha)
#     #
#     return np.array([delta, theta, alpha, beta, gamma])

def my_plot(rest_arr, arithmetic_arr, n):
    rest_mean = np.mean(rest_arr, axis=0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    for i, row in enumerate(rest_arr):
        plt.plot(row, color='lightgray', alpha=0.6, linewidth=0.8)
    plt.plot(rest_mean, color='red', label='Mean', linewidth=2)
    # 添加图例（可选）
    plt.legend()  # 图例显示行号

    # 设置标题和坐标轴标签
    plt.title(f"Line Plot for Each Trial of Rest for Subject {n}")
    plt.xlabel("Index of Data Points")
    plt.ylabel("Values")

    # 显示图形
    plt.show()



    arithmetic_mean = np.mean(arithmetic_arr, axis=0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    for i, row in enumerate(arithmetic_arr):
        plt.plot(row, color='lightgray', alpha=0.6, linewidth=0.8)
    plt.plot(arithmetic_mean, color='red', label='Mean', linewidth=2)
    # 添加图例（可选）
    plt.legend()  # 图例显示行号

    # 设置标题和坐标轴标签
    plt.title(f"Line Plot for Each Trial of Arithmetic for subject {n}")
    plt.xlabel("Index of Data Points")
    plt.ylabel("Values")

    # 显示图形
    plt.show()


def plot_all_trial_attention(all_trial_arr, n, current_window_data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    plt.plot(all_trial_arr, color='red', label='all_trial', linewidth=2)
    plt.plot((current_window_data.events[:, -1]//16) -1, color='blue', label='event', linewidth=2)
    # 添加图例（可选）
    plt.legend()  # 图例显示行号

    # 设置标题和坐标轴标签
    plt.title(f"Line Plot for A single session of Attention for Subject {n}")
    plt.xlabel("Index of Data Points")
    plt.ylabel("Values")

    plt.show()


def plot_two_class(rest_arr, arithmetic_arr):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))

    plt.plot(rest_arr, color='blue', label='rest', linewidth=2)
    plt.plot(arithmetic_arr, color='red', label='arithmetic', linewidth=2)

    plt.legend()
    plt.xlabel("Index of Trial")
    plt.ylabel("Attention Value")

    plt.show()

if __name__ == "__main__":
    count = 29
    # FRONTAL_CHANNEL = ['F7', 'FAF5', 'F3', 'AFp1', 'FAF1', 'AFp2', 'FAF2', 'FAF6', 'F4', 'F8']
    # FRONTAL_CHANNEL = ['F7', 'AFF5h', 'F3', 'AFp1', 'AFF1h', 'AFp2', 'AFF2h', 'AFF6h', 'F4', 'F8']
    FRONTAL_CHANNEL = ['AFp1']  # 'AFF1h', 'AFp2', 'AFF2h'
    PARIETAL_CHANNEL = ['Pz', 'P3', 'P4', 'P7', 'P8']

    MOTOR_CHANNEL = ['CFC5', 'CFC6', 'CFC3', 'CFC4', 'Cz,', 'CCP5', 'CCP6', 'CCP3', 'CCP4']
    OCCIPITAL_CHANNEL = ['PPO1', 'OPO1', 'OPO2', 'PPO2']

    SELECTED_CHANNELS = FRONTAL_CHANNEL

    config = {
        'is_MA': True,
        'down_sampling_rate': 200,

        'epoch_start': -5,
        'epoch_end': 25,

        'baseline': (-3.0, -0.05),  # base of each epoch is from -3s to 0s

        'filter': {
            'MA': {
                'Wp': (0.0400, 0.3000),  # pass band (0.04*200/2)=4 to (0.35*200/2)=35
                'Ws': (0.0100, 0.3500),  # stop band edge
                'Rp': 3,  # pass band ripple
                'Rs': 30,  # stop band ripple
            }
        },

        'windows': range(-5, 25),
        'window_duration': 3.0  # 0+3, 1+3, 2+3, 3+3, 4+3, 5+3, 6+3, 7+3, 8+3, 9+3

    }

    raw_data_path = r'D:\dataset\Open_Access_Dataset_for_EEG_NIRS_Single-Trial_Classification\official_release\subject '

    events = dict()
    events.update({'subtraction': 3, 'rest': 4})

    paradigms = 'arithmetic'
    n_sessions = 3

    # data = loadmat(fname, squeeze_me=True, struct_as_record=False)

    rest_arr_all = []
    arithmetic_arr_all = []

    data_records = []

    for n in range(1, count+1):  # count+1  ############## per subject ##############
        subject = str(n) if n>=10 else '0'+str(n)
        data_ = loadmat(raw_data_path + subject + '\\' + r'with occular artifact\cnt.mat', squeeze_me=True, struct_as_record=False)
        mark_ = loadmat(raw_data_path + subject + '\\' + r'with occular artifact\mrk.mat', squeeze_me=True, struct_as_record=False)
        mnt_ = loadmat(raw_data_path + subject + '\\' + r'with occular artifact\mnt.mat', squeeze_me=True, struct_as_record=False)

        data = data_['cnt']
        mark = mark_['mrk']
        mnt = mnt_['mnt']

        # MA: arithmetic and rest
        MA_sessions = [1,3,5]

        subject_data = dict()
        for session_num in MA_sessions:
            _ = convert_one_session(data, mark, session_num)
            subject_data[f'subject{n}_session{session_num}_arithmetic'] = _
        # subject_data is a dict of shape:
        # {'1arithmetic': {'0': < RawArray | 33 x 120236(601.2 s), ~30.3MB, data loaded >},
        #  '3arithmetic': {'0': < RawArray | 33 x 120451(602.2s), ~30.4MB, data loaded >},
        #  '5arithmetic': {'0': < RawArray | 33 x 118853(594.3s), ~30.0MB, data loaded >}
        # }


        # extract epochs
        subject_data = extract_epochs(subject_data, config)  # in shape of (60, 15, 6000)

        windows = config['windows']

        window_att = []
        for window_start in windows:  ############## per window ############
            # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # len(windows) = 15

            # windows_data = get_window_data(subject_data, config, window_start)

            # example:
            # window_start = 0.0
            # window_duration = 3.0
            # window_end = 2.995
            # total points = 600
            # crop from 0.0 to 2.995, including 0.0 and 2.995
            window_end = window_start + config['window_duration'] - 1./config['down_sampling_rate']
            current_window_data = subject_data.copy().crop(window_start, window_end)  # current_window_data in shape of (60, 15, 600)


            # TODO 得到窗口数据(60,15,600)，开始计算attention值
            trial_att = compute_(current_window_data)  # (60,)
            window_att.append(trial_att)
        _ = np.stack(window_att, axis=1)  # _ in shape of (60, 10)

        # compute average for both arithmetic(3) and rest(4)
        rest_arr = []
        arithmetic_arr = []
        for index, event_type in enumerate(subject_data.events[:, -1]):
            if event_type == 32:
                # rest
                rest_arr.append(_[index])
            elif event_type == 16:
                # arithmetic
                arithmetic_arr.append(_[index])
        rest_arr = np.stack(rest_arr, axis=0)  # (30, 1)
        arithmetic_arr = np.stack(arithmetic_arr, axis=0)  # (30, 1)

        plot_all_trial_attention(_, n, current_window_data)
        # plot_two_class(rest_arr, arithmetic_arr)

        # my_plot(rest_arr, arithmetic_arr, n)

        # rest_att_average = np.mean(rest_arr, axis=0)
        # arithmetic_att_average = np.mean(arithmetic_arr, axis=0)

        # rest_avg = np.mean(rest_att_average)
        # arithmetic_avg = np.mean(arithmetic_att_average)

        rest_arr_all.append(rest_arr)
        arithmetic_arr_all.append(arithmetic_arr)

    rest_arr_all = np.vstack(rest_arr_all)
    arithmetic_arr_all = np.vstack(arithmetic_arr_all)

    a = np.std(rest_arr_all, axis=0)
    b = np.std(arithmetic_arr_all, axis=0)


    pass

    #         temp_data = data[0][column]['x'][0][0]
    #         temp_mark = mark[0][column]['time'][0][0]
    #         trail_num = temp_mark.shape[1]
    #
    #     pass


