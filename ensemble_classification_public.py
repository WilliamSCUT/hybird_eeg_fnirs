import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EnsembleAttentionClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        # 只使用随机森林分类器
        self.rf = RandomForestClassifier(n_estimators=200, random_state=42)
        
    def load_data(self, data_path):
        """加载处理好的特征数据"""
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        # 加载配置
        self.config = data['config'].item()
        
        # 将标签16和32转换为0和1
        y = (y == 32).astype(int)  # 16->0, 32->1
        
        # 展平X的前两个维度并重复y
        n_windows = X.shape[1]
        X_reshaped = X.reshape(-1, X.shape[-1])
        y_repeated = np.repeat(y, n_windows)
        
        return X_reshaped, y_repeated
        
    def train(self, X, y):
        """训练随机森林分类器"""
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练随机森林
        self.rf.fit(X_train_scaled, y_train)
        
        # 评估模型
        y_pred = self.rf.predict(X_test_scaled)
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_test_scaled, y_test
        
    def calculate_attention_score(self, X):
        """计算注意力分数(0-100)"""
        # X已经是2D的了，直接使用
        X_scaled = self.scaler.transform(X)
        
        # 获取随机森林的概率预测
        rf_proba = self.rf.predict_proba(X_scaled)[:, 1]
        
        # 转换为0-100分数
        scores = rf_proba * 100
        
        return scores
        
    def analyze_scores(self, X, y):
        """统计分析分数分布"""
        scores = self.calculate_attention_score(X)
        
        # 确保y是一维数组
        y = y.ravel()
        
        # 分别获取两类样本的分数
        low_attention_scores = scores[y == 0]
        high_attention_scores = scores[y == 1]
        
        # 打印调试信息
        print(f"Number of low attention samples: {len(low_attention_scores)}")
        print(f"Number of high attention samples: {len(high_attention_scores)}")
        
        # 进行t检验
        t_stat, p_value = stats.ttest_ind(low_attention_scores, high_attention_scores)
        
        # 绘制分布图
        plt.figure(figsize=(12, 6))
        
        # 绘制核密度估计
        plt.subplot(1, 2, 1)
        sns.kdeplot(data=low_attention_scores, label='Rest (Low Attention)')
        sns.kdeplot(data=high_attention_scores, label='Arithmetic (High Attention)')
        plt.xlabel('Attention Score')
        plt.ylabel('Density')
        plt.title('Distribution of Attention Scores')
        plt.legend()
        
        # 绘制箱线图
        plt.subplot(1, 2, 2)
        data = [low_attention_scores, high_attention_scores]
        plt.boxplot(data, labels=['Rest', 'Arithmetic'])
        plt.ylabel('Attention Score')
        plt.title('Attention Score Distribution by Class')
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print("\nStatistical Analysis:")
        print(f"Low Attention (Rest) - Mean: {np.mean(low_attention_scores):.2f}, Std: {np.std(low_attention_scores):.2f}")
        print(f"High Attention (Arithmetic) - Mean: {np.mean(high_attention_scores):.2f}, Std: {np.std(high_attention_scores):.2f}")
        print(f"T-statistic: {t_stat:.2f}")
        print(f"P-value: {p_value:.4f}")
        
        return {
            'low_attention_stats': {
                'mean': np.mean(low_attention_scores),
                'std': np.std(low_attention_scores),
                'median': np.median(low_attention_scores)
            },
            'high_attention_stats': {
                'mean': np.mean(high_attention_scores),
                'std': np.std(high_attention_scores),
                'median': np.median(high_attention_scores)
            }
        }

if __name__ == "__main__":
    # 使用示例
    classifier = EnsembleAttentionClassifier()
    
    # 加载所有受试者的数据
    all_X = []
    all_y = []
    
    for subject in range(1, 30):
        try:
            data_path = f'processed_data_2/subject_{subject}_features.npz'
            X, y = classifier.load_data(data_path)
            all_X.append(X)
            all_y.append(y)
        except FileNotFoundError:
            print(f"Warning: Data not found for subject {subject}")
            continue
    
    # 合并所有数据
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # 训练模型
    X_test, y_test = classifier.train(X, y)
    
    # 分析分数分布
    stats_results = classifier.analyze_scores(X, y) 