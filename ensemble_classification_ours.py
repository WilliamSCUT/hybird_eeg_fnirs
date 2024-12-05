import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import joblib


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
        
        # 打印数据形状
        print(f"Loaded data shapes:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # 加载配置
        self.config = data['config'].item()
        
        # 将数据展平成二维数组
        X = X.reshape(X.shape[0], -1)
        
        return X, y
        
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
        
        # 返回测试集数据和标签
        return X_train, X_test, y_train, y_test  # 添加返回训练集数据
        
    def calculate_attention_score(self, X):
        """计算注意力分数(0-100)"""
        X_scaled = self.scaler.transform(X)
        rf_proba = self.rf.predict_proba(X_scaled)[:, 1]
        scores = rf_proba * 100
        
        # 添加调试信息
        print("\nScore Statistics:")
        print(f"Min score: {np.min(scores):.2f}")
        print(f"Max score: {np.max(scores):.2f}")
        print(f"Mean score: {np.mean(scores):.2f}")
        
        # 确保分数在0-100范围内
        scores = np.clip(scores, 0, 100)
        
        return scores
        
    def analyze_scores(self, X, y, output_dir, feature_file_name):
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
        
        # 调试信息
        print("\nDebug: Saving plot...")
        print(f"Output directory: {output_dir}")
        print(f"Plot filename: {feature_file_name}_attention_scores.png")
        
        # 检查目录是否存在
        if not os.path.exists(output_dir):
            print(f"Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # 完整的保存路径
        plot_filename = os.path.join(output_dir, f"{feature_file_name}_attention_scores.png")
        print(f"Full save path: {plot_filename}")
        
        # 尝试保存图片
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print("Plot saved successfully!")
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
        

        # 保存前显示图片（如果在交互环境中）
        plt.show()
        # 确保关闭图片以释放内存
        plt.close()
        
        # Save statistical results
        stats_filename = os.path.join(output_dir, f"{feature_file_name}_stats.txt")
        with open(stats_filename, 'w') as f:
            f.write("\nStatistical Analysis:\n")
            f.write(f"Low Attention (Rest) - Mean: {np.mean(low_attention_scores):.2f}, Std: {np.std(low_attention_scores):.2f}\n")
            f.write(f"High Attention (Arithmetic) - Mean: {np.mean(high_attention_scores):.2f}, Std: {np.std(high_attention_scores):.2f}\n")
            f.write(f"T-statistic: {t_stat:.2f}\n")
            f.write(f"P-value: {p_value:.4f}\n")

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

    def save_model(self, filename):
        """保存模型"""
        joblib.dump(self.rf, filename)

    def get_model(self):
        return self.rf
if __name__ == "__main__":
    # 使用示例
    feature_files = [
        'preprocessed_data/features/features_subject_03.npz',
        'preprocessed_data/features/features_subject_04.npz',
        'preprocessed_data/features/features_subject_05.npz',
        'preprocessed_data/features/features_subject_06.npz',
        'preprocessed_data/features/features_subject_07.npz',
        'preprocessed_data/features/features_subject_08.npz',
        'preprocessed_data/features/features_subject_09.npz',
        'preprocessed_data/features/features_subject_10.npz',
        'preprocessed_data/features/features_subject_11.npz',
    ]
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # 为每个特征文件创建分类器并评估
    for feature_file in feature_files:
        print(f"\n\nProcessing {feature_file}:")
        print("="*50)
        
        classifier = EnsembleAttentionClassifier()
        
        # 加载特征数据
        X, y = classifier.load_data(feature_file)
        
        # 训练模型
        X_train, X_test, y_train, y_test = classifier.train(X, y)
        
        # 分析全部数据分数分布
        # stats_results = classifier.analyze_scores(X, y, output_dir, os.path.basename(feature_file))

        # 分析测试集分数分布
        stats_results = classifier.analyze_scores(X_test, y_test, output_dir, os.path.basename(feature_file))
        
        # 打印统计结果
        print("\nSummary Statistics:")
        print(f"Low Attention Mean: {stats_results['low_attention_stats']['mean']:.2f}")
        print(f"High Attention Mean: {stats_results['high_attention_stats']['mean']:.2f}")

    # Combine all feature files for a single training session
    print("\n\nProcessing combined features:")
    print("="*50)
    
    combined_X, combined_y = [], []
    for feature_file in feature_files:
        classifier = EnsembleAttentionClassifier()
        X, y = classifier.load_data(feature_file)
        combined_X.append(X)
        combined_y.append(y)
    
    combined_X = np.vstack(combined_X)
    combined_y = np.hstack(combined_y)
    
    classifier = EnsembleAttentionClassifier()
    X_train, X_test, y_train, y_test = classifier.train(combined_X, combined_y)



    model_filename = os.path.join(output_dir, 'combined_model.pkl')
    classifier.save_model(model_filename)

    # Convert into ONNX format with explicit type specification
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    # 指定输入类型为 float32
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onx = to_onnx(classifier.get_model(), X[:1].astype(np.float32), initial_types=initial_type)
    
    model_filename = os.path.join(output_dir, 'combined_model.onnx')
    with open(model_filename, "wb") as f:
        f.write(onx.SerializeToString())

    # 使用 ONNX Runtime 进行推理
    import onnxruntime as rt

    sess = rt.InferenceSession(model_filename, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # 确保输入数据类型为 float32
    X_test_float32 = X_test.astype(np.float32)
    pred_onx = sess.run([label_name], {input_name: X_test_float32})[0]

    
    # 分析全部数据分数分布
    # stats_results = classifier.analyze_scores(combined_X, combined_y, output_dir, "combined")

    # 分析测试集分数分布
    stats_results = classifier.analyze_scores(X_test, y_test, output_dir, "combined")
    
    # 打印统计结果
    print("\nSummary Statistics for Combined Features:")
    print(f"Low Attention Mean: {stats_results['low_attention_stats']['mean']:.2f}")
    print(f"High Attention Mean: {stats_results['high_attention_stats']['mean']:.2f}")