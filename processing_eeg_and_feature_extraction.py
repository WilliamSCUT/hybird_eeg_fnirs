import numpy as np
import mne
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
import os

FRONTAL_CHANNEL = ['AFp1', 'AFF1h', 'AFp2', 'AFF2h']
PARIETAL_CHANNEL = ['Pz', 'P3', 'P4', 'P7', 'P8']
MOTOR_CHANNEL = ['CFC5', 'CFC6', 'CFC3', 'CFC4', 'Cz,', 'CCP5', 'CCP6', 'CCP3', 'CCP4']
OCCIPITAL_CHANNEL = ['PPO1', 'OPO1', 'OPO2', 'PPO2']



SELECTED_CHANNELS = FRONTAL_CHANNEL

class EEGProcessor:
    def __init__(self, config):
        self.config = config
        self.csp = CSP(n_components=4, reg=None, log=True)
        self.scaler = StandardScaler()
        
    def extract_frequency_bands(self, data, fs):
        """提取频带能量特征"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        features = []
        for band_name, (low, high) in bands.items():
            # 带通滤波
            b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
            filtered = signal.filtfilt(b, a, data)
            # 计算能量
            power = np.mean(filtered**2, axis=-1)
            features.append(power)
            
        return np.array(features)

    def extract_connectivity_features(self, data):
        """提取连接性特征"""
        from scipy.signal import coherence
        n_channels = data.shape[0]
        connectivity = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                f, coh = coherence(data[i], data[j])
                connectivity[i,j] = np.mean(coh)
                connectivity[j,i] = connectivity[i,j]
                
        return connectivity.flatten()

    def extract_time_features(self, data):
        """提取时域特征"""
        features = []
        # 统计特征
        features.append(np.mean(data, axis=-1))
        features.append(np.std(data, axis=-1))
        features.append(np.max(data, axis=-1))
        features.append(np.min(data, axis=-1))
        # Hjorth特征
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)
        mobility = np.std(diff1, axis=-1) / np.std(data, axis=-1)
        complexity = (np.std(diff2, axis=-1) * np.std(data, axis=-1)) / (np.std(diff1, axis=-1)**2)
        features.extend([mobility, complexity])
        
        return np.array(features)

    def process_single_trial(self, trial_data, fs, label=None):
        """处理单个试次的数据"""
        # 1. 频带能量特征
        freq_features = self.extract_frequency_bands(trial_data, fs)
        
        # 2. 连接性特征
        conn_features = self.extract_connectivity_features(trial_data)
        
        # 3. 时域特征
        time_features = self.extract_time_features(trial_data)
        
        # 合并所有特征
        all_features = np.concatenate([
            freq_features.flatten(),
            conn_features,
            time_features.flatten()
        ])
        
        return all_features

    def process_data(self, raw_data_path, subject_range=(1, 30)):
        """处理所有受试者的数据"""
        for n in tqdm(range(*subject_range), desc="Processing subjects"):
            subject = str(n) if n>=10 else '0'+str(n)
            
            # 加载数据
            data_ = loadmat(f"{raw_data_path}{subject}/with occular artifact/cnt.mat", 
                          squeeze_me=True, struct_as_record=False)
            mark_ = loadmat(f"{raw_data_path}{subject}/with occular artifact/mrk.mat", 
                          squeeze_me=True, struct_as_record=False)
            
            # 提取epochs和处理数据
            epochs = self.extract_epochs(data_['cnt'], mark_['mrk'])
            
            # 特征提取
            X, y = self.extract_features(epochs)
            
            # 添加保存信息的打印
            print(f"\nSaving features for subject {n}:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            
            # 保存处理后的数据
            save_path = f'processed_data_2/subject_{n}_features.npz'
            np.savez(save_path,
                    X=X,
                    y=y,
                    channel_names=SELECTED_CHANNELS,
                    config=self.config)
            
            # 确认文件已保存
            print(f"Features saved to: {save_path}")

    def extract_epochs(self, data, mark):
        """从原始数据中提取epochs"""
        # 提取MA sessions
        MA_sessions = [1, 3, 5]
        
        # 处理每个session
        subject_data = dict()
        for session_num in MA_sessions:
            _ = self.convert_one_session(data, mark, session_num)
            subject_data[f'session{session_num}_arithmetic'] = _
        
        # 提取epochs的参数
        fps = self.config['down_sampling_rate']
        duration_of_epoch = int((self.config['epoch_end'] - self.config['epoch_start']) * fps) + 1
        
        # 获取原始info并创建新的只包含选定通道的info
        raw = list(subject_data.values())[0]['0']
        picks = [raw.ch_names.index(ch) for ch in SELECTED_CHANNELS]
        info = mne.pick_info(raw.info, picks)
        
        # 准备存储空间
        instance_data = np.empty((60, len(SELECTED_CHANNELS), duration_of_epoch))
        instance_events = np.empty((3, 20, 3), dtype=np.int32)
        instance_durations = np.empty((3,), dtype=np.int32)
        
        # 处理每个session的数据
        for i, session_data in enumerate(subject_data.values()):
            raw = session_data['0']
            raw.set_eeg_reference(ref_channels='average')
            
            # 选择通道
            raw.pick_channels(SELECTED_CHANNELS + ['Stim'])
            
            # 应用滤波器
            raw.filter(self.config['filter']['bandpass']['low'],
                      self.config['filter']['bandpass']['high'],
                      picks=['eeg'])
            
            if self.config['filter'].get('notch'):
                raw.notch_filter(self.config['filter']['notch'],
                               picks=['eeg'])
            
            # 提取trials
            events = mne.find_events(raw, stim_channel='Stim')
            
            # 创建epochs
            epochs = mne.Epochs(raw, events, 
                              tmin=self.config['epoch_start'],
                              tmax=self.config['epoch_end'],
                              baseline=self.config['baseline'],
                              picks=SELECTED_CHANNELS,
                              preload=True)
            
            # 存储据
            instance_data[i*20:(i+1)*20] = epochs.get_data()
            instance_events[i] = epochs.events
            instance_durations[i] = len(raw.times)
        
        # 调整events的时间点
        instance_events[1, :, 0] += instance_durations[0]
        instance_events[2, :, 0] += np.sum(instance_durations[0:2])
        instance_events = instance_events.reshape((60, 3))
        
        # 创建最终的epochs对象，使用更新后的info
        epochs = mne.EpochsArray(instance_data, info,
                                events=instance_events,
                                tmin=self.config['epoch_start'],
                                baseline=self.config['baseline'])
        
        return epochs

    def extract_features(self, epochs):
        """从epochs中提取特征"""
        features = []
        labels = []
        
        # 获取所有trials的数据
        data = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
        events = epochs.events
        
        # 首先对整个数据集进行CSP拟合
        self.csp.fit(data, events[:, -1])
        
        for trial_idx in range(len(data)):
            # 对每个时间窗口进行处理
            window_features = []
            for window_start in self.config['windows']:
                window_end = window_start + self.config['window_duration']
                
                # 计算窗口的采样点索引
                start_idx = int((window_start - self.config['epoch_start']) * self.config['down_sampling_rate'])
                end_idx = int((window_end - self.config['epoch_start']) * self.config['down_sampling_rate'])
                
                # 提取窗口数据
                window_data = data[trial_idx, :, start_idx:end_idx]
                
                # 使用已训练的CSP进行变换
                window_csp = self.csp.transform(window_data[np.newaxis, :, :])
                
                # 处理每个试次
                trial_features = self.process_single_trial(
                    window_data[np.newaxis, :, :],
                    self.config['down_sampling_rate']
                )
                
                # 添加CSP特征
                window_features.append(np.concatenate([
                    trial_features,
                    window_csp.flatten()  # CSP特征已经是2D的了
                ]))
            
            features.append(window_features)
            labels.append(events[trial_idx, -1])
        
        # 添加调试信息
        print(f"Features shape: {np.array(features).shape}")
        print(f"Labels shape: {np.array(labels).shape}")
        
        return np.array(features), np.array(labels)

    def convert_one_session(self, data, mark, session_num):
        """转换单个session的数据为MNE格式"""
        eeg = data[session_num].x.T * 1e-6  # from uV to V
        trig = np.zeros((1, eeg.shape[1]))
        
        # 降采样到200Hz的索引
        idx = (mark[session_num].time - 1) // 5
        trig[0, idx] = mark[session_num].event.desc
        
        eeg = np.vstack([eeg, trig])
        channel_name = list(data[session_num].clab) + ['Stim']
        channel_types = ['eeg'] * 30 + ['eog'] * 2 + ['stim']
        
        # 创建MNE Raw对象
        montage = mne.channels.make_standard_montage('standard_1005')
        info = mne.create_info(channel_name, self.config['down_sampling_rate'], channel_types)
        raw = mne.io.RawArray(eeg, info, verbose=False)
        raw.set_montage(montage)
        
        return {'0': raw}

if __name__ == "__main__":
    config = {
        'down_sampling_rate': 200,
        'epoch_start': -5,
        'epoch_end': 25,
        'baseline': (-3.0, -0.05),
        'windows': range(-5, 10),
        'window_duration': 3.0,
        'filter': {
            'bandpass': {
                'low': 0.5,
                'high': 45,
            },
            'notch': 50,
        }
    }
    
    # 创建保存目录
    os.makedirs('processed_data_2', exist_ok=True)
    
    # 设置数据路径
    raw_data_path = r'C:\Users\lizhi\Desktop\dataset\EEG\subject '
    
    processor = EEGProcessor(config)
    processor.process_data(raw_data_path) 