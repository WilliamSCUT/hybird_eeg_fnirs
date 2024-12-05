import numpy as np
from scipy import signal
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

class OurEEGProcessor:
    def __init__(self, config):
        self.config = config
        self.csp = CSP(n_components=4, reg=None, log=True)
        self.scaler = StandardScaler()
        
    def extract_frequency_bands(self, data, fs):
        """Extract frequency band features"""
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        features = []
        for band_name, (low, high) in bands.items():
            b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
            filtered = signal.filtfilt(b, a, data)
            power = np.mean(filtered**2, axis=-1)
            features.append(power)
            
        return np.array(features)

    def extract_connectivity_features(self, data):
        """Extract connectivity features"""
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
        """Extract time domain features"""
        features = []
        features.append(np.mean(data, axis=-1))
        features.append(np.std(data, axis=-1))
        features.append(np.max(data, axis=-1))
        features.append(np.min(data, axis=-1))
        
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)
        mobility = np.std(diff1, axis=-1) / np.std(data, axis=-1)
        complexity = (np.std(diff2, axis=-1) * np.std(data, axis=-1)) / (np.std(diff1, axis=-1)**2)
        features.extend([mobility, complexity])
        
        return np.array(features)

    def process_single_trial(self, trial_data, fs):
        """Process a single trial"""
        freq_features = self.extract_frequency_bands(trial_data, fs)
        conn_features = self.extract_connectivity_features(trial_data)
        time_features = self.extract_time_features(trial_data)
        
        all_features = np.concatenate([
            freq_features.flatten(),
            conn_features,
            time_features.flatten()
        ])
        
        return all_features

    def process_data(self, subject_ids, data_dir, save_dir, ma_file_pattern, rest_file_pattern):
        """Process data for one or multiple subjects
        Args:
            subject_ids: List of subject IDs to process
            data_dir: Base directory containing the raw data
            save_dir: Directory to save processed features
            ma_file_pattern: File pattern for MA data
            rest_file_pattern: File pattern for REST data
        """
        all_X = []
        all_y = []
        all_subjects = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        for subject_id in subject_ids:
            print(f"\nProcessing subject {subject_id}")
            
            # Construct file paths for this subject
            ma_file = os.path.join(data_dir, f'subject_{subject_id:02d}', 'preprocessed_data', ma_file_pattern)
            rest_file = os.path.join(data_dir, f'subject_{subject_id:02d}', 'preprocessed_data', rest_file_pattern)
            
            if not os.path.exists(ma_file) or not os.path.exists(rest_file):
                print(f"Warning: Files for subject {subject_id} not found, skipping...")
                continue
            
            # Load MA (Mental Arithmetic) data
            ma_data = np.load(ma_file)
            ma_trials = ma_data['data']
            ma_labels = ma_data['labels']  # 原始代码
            
            # Load REST data
            rest_data = np.load(rest_file)
            rest_trials = rest_data['data']
            rest_labels = np.zeros_like(rest_data['labels'])  # 原始代码
            
            # # 新增：随机翻转一半的标签
            # ma_labels = ma_data['labels'].copy()
            # rest_labels = np.zeros_like(rest_data['labels'])
            
            # # 随机选择一半的MA标签进行翻转
            # flip_indices = np.random.choice(len(ma_labels), size=len(ma_labels)//2, replace=False)
            # ma_labels[flip_indices] = 0  # 将选中的MA标签改为0
            
            # # 随机选择一半的REST标签进行翻转
            # rest_flip_indices = np.random.choice(len(rest_labels), size=len(rest_labels)//2, replace=False)
            # rest_labels[rest_flip_indices] = 1  # 将选中的REST标签改为1
            
            # 添加调试信息来检查标签
            print("\nMA Labels distribution (after flipping):")
            print(f"Unique labels: {np.unique(ma_labels, return_counts=True)}")
            
            print("\nREST Labels distribution (after flipping):")
            print(f"Unique labels: {np.unique(rest_labels, return_counts=True)}")
            
            
            # Combine all trials and labels
            all_trials = np.concatenate([ma_trials, rest_trials], axis=0)
            all_labels = np.concatenate([ma_labels, rest_labels], axis=0)
            
            # Debug info for window calculations
            print("\nWindow calculations:")
            for window_start in self.config['windows']:
                window_end = window_start + self.config['window_duration']
                start_idx = int(window_start * self.config['sampling_rate'])
                end_idx = int(window_end * self.config['sampling_rate'])
                print(f"Window {window_start}-{window_end}s: samples {start_idx}-{end_idx}")
            
            # Process each trial
            features_list = []
            
            for trial_idx in tqdm(range(len(all_trials)), desc=f"Processing subject {subject_id}"):
                trial_data = all_trials[trial_idx]
                
                # Process windows within the trial
                window_features = []
                for window_start in self.config['windows']:
                    window_end = window_start + self.config['window_duration']
                    
                    # Convert time to samples
                    start_idx = int(window_start * self.config['sampling_rate'])
                    end_idx = int(window_end * self.config['sampling_rate'])
                    
                    # Extract window data
                    window_data = trial_data[:, start_idx:end_idx]
                    
                    # Extract features for this window
                    trial_features = self.process_single_trial(
                        window_data,
                        self.config['sampling_rate']
                    )
                    window_features.append(trial_features)
                
                features_list.append(window_features)
            
            X = np.array(features_list)
            y = all_labels
            
            all_X.append(X)
            all_y.append(y)
            all_subjects.extend([subject_id] * len(y))
            
            # Save individual subject data
            subject_save_path = os.path.join(save_dir, f'features_subject_{subject_id:02d}.npz')
            np.savez(subject_save_path,
                    X=X, y=y, subject_id=subject_id,
                    channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
                    config=self.config)
            
            print(f"\nFeatures saved for subject {subject_id}: {subject_save_path}")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
        
        # Save combined data
        combined_save_path = os.path.join(save_dir, 'features_all_subjects.npz')
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        subjects_combined = np.array(all_subjects)
        
        np.savez(combined_save_path,
                X=X_combined, y=y_combined, subjects=subjects_combined,
                channel_names=['CH1', 'CH2', 'CH3', 'CH4'],
                config=self.config)
        
        print("\nProcessing complete!")
        print(f"Combined data saved to: {combined_save_path}")
        print(f"Total X shape: {X_combined.shape}")
        print(f"Total y shape: {y_combined.shape}")
        print(f"Subjects processed: {sorted(set(all_subjects))}")
        

if __name__ == "__main__":
    config = {
        'sampling_rate': 256,
        'windows': [0],  # 只使用完整的3秒数据
        'window_duration': 3.0,  # 保持3秒窗口长度
    }
    
    # 定义路径和受试者ID
    data_dir = 'preprocessed_data'  # 原始数据的基础目录
    save_dir = 'preprocessed_data/features'  # 特征保存目录
    subject_ids = [3,4,5,6,7,8,9,10,11]  # 可以改为 [3, 4, 5] 来处理多个受试者
    
    # 文件名模式
    ma_file_pattern = 'trial-160_channel-4_epoch-0.0-10.0_fs-256.npz'
    rest_file_pattern = 'trial-160_channel-4_epoch-0.0-10.0_fs-256_REST.npz'
    
    processor = OurEEGProcessor(config)
    processor.process_data(subject_ids, data_dir, save_dir, ma_file_pattern, rest_file_pattern) 