import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib
matplotlib.use('Agg')

# 设置 matplotlib 使用支持中文的字体
plt.rcParams['font.family'] = ['SimHei']  # Windows系统可以使用'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def preprocess_data(X, y):
    """对特征和标签进行预处理"""
    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 特征归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, label_encoder, scaler

def train_model(X_train, y_train):
    """训练XGBoost模型"""
    clf_best = XGBClassifier(n_estimators=100, max_depth=3, min_child_weight=1, gamma=0.2,
                             subsample=0.7, colsample_bytree=0.6, learning_rate=0.1, reg_lambda=5,
                             random_state=42)
    clf_best.fit(X_train, y_train)
    return clf_best

def predict_and_visualize(model, scaler, label_encoder, features):
    """使用模型进行预测并可视化结果"""
    # 将输入的特征值归一化
    features_scaled = scaler.transform(features)

    # 使用模型进行预测
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)

    # 输出预测结果
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # 将预测结果翻译为中文
    if predicted_class == 'f':
        predicted_class_chinese = '弯曲破坏'
    elif predicted_class == 'fs':
        predicted_class_chinese = '弯剪破坏'
    elif predicted_class == 's':
        predicted_class_chinese = '剪切破坏'
    else:
        predicted_class_chinese = '未知破坏模式'  # 针对其他未知类别

    # 输出预测概率的直方图（每个条形使用不同的颜色）
    colors = ['blue', 'green', 'red']
    plt.bar(label_encoder.classes_, prediction_proba[0], color=colors)
    plt.xlabel('破坏模式')
    plt.ylabel('预测概率')
    plt.title('墩柱破坏模式预测概率')
    plt.savefig('myapp/static/myapp/prediction.png')  # 保存图形到静态目录
    plt.close()

    return predicted_class_chinese, {label_encoder.classes_[i]: prob for i, prob in enumerate(prediction_proba[0])}

def explain_prediction(model, scaler, label_encoder, features):
    """计算并展示特定样本的SHAP值解释，并输出SHAP值表格"""
    # 应用与训练数据相同的预处理步骤
    features_scaled = scaler.transform(features)

    # 计算新样本的SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)

    # 使用字典来收集SHAP值
    shap_values_dict = {}

    # 遍历每个样本和每个类别，画出SHAP force_plot，并收集SHAP值
    for i in range(len(features_scaled)):
        for j in range(len(label_encoder.classes_)):
            class_name = label_encoder.classes_[j]
            print(f"对于破坏模式 '{class_name}':")
            shap.force_plot(
                base_value=explainer.expected_value[j],
                shap_values=shap_values[j][i],
                features=features.iloc[i],
                feature_names=features.columns,
                matplotlib=True
            )
            plt.gcf().set_size_inches(12, 3)  # 调整当前图形的大小
            plt.savefig(f'myapp/static/myapp/shap_force_plot_sample_{i + 1}_class_{class_name}.png',
                        bbox_inches='tight')  # 保存图形
            plt.close()

            # 将当前类别的SHAP值添加到字典中
            shap_values_dict[class_name] = shap_values[j][i]

    # 将字典转换为DataFrame
    shap_values_df = pd.DataFrame(shap_values_dict, index=features.columns)

    # 绘制SHAP值条形图
    shap_values_df.plot(kind='bar', figsize=(12, 8))
    plt.title('各特征对应各破坏模式的SHAP值')
    plt.xlabel('破坏模式')
    plt.ylabel('SHAP值')
    plt.xticks(rotation=45)
    plt.legend(title='特征', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('myapp/static/myapp/shap_values.png')  # 保存图形
    plt.close()

    return shap_values_df
