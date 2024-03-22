from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib.auth.models import User
from .utils import predict_and_visualize, preprocess_data, train_model, explain_prediction
import pandas as pd
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from django.db import IntegrityError  # 引入IntegrityError


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            # 登录成功，重定向到index页面
            return redirect('index')  # 确保这里的 'index' 是index视图的URL名称
        else:
            # 登录失败，显示错误
            return render(request, 'login.html', {'error': '无效的用户名或密码'})

    return render(request, 'login.html')

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        # 可以添加更多字段，如email等

        try:
            # 创建新用户
            User.objects.create_user(username=username, password=password)

            # 自动登录新用户
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('index')  # 重定向到index页面
        except IntegrityError:
            # 如果用户名已存在，则捕捉到异常
            context = {'error': '用户名已存在，请选择其他用户名。'}
            return render(request, 'register.html', context)

    return render(request, 'register.html')

def logout_view(request):
    logout(request)
    # 这里假设 'login' 是登录页面的 URL 名称
    return redirect('login')

@login_required
def index(request):
    return render(request, 'myapp/index.html')


def predict(request):
    if request.method == 'POST':
        # 从表单获取用户输入
        nsv = float(request.POST.get('nsv'))
        al = float(request.POST.get('al'))
        av = float(request.POST.get('av'))
        n = float(request.POST.get('n'))
        λ = float(request.POST.get('λ'))
        sh0 = float(request.POST.get('sh0'))
        print("Request POST parameters:", request.POST.dict())


        # 创建样本 DataFrame
        sample = pd.DataFrame({
            "nsv": [nsv],
            "al": [al],
            "av": [av],
            "n": [n],
            "λ": [λ],
            "s/h0": [sh0]
        })

        # 加载数据集
        data = pd.read_csv("data/column_墩柱破坏模式.csv")

        # 创建特征列和目标列
        X = data[["nsv", "al", "av", "n", "λ", "s/h0"]]
        y = data["result"]

        # 数据预处理
        X_scaled, y_encoded, label_encoder, scaler = preprocess_data(X, y)

        # 训练模型
        clf_best = train_model(X_scaled, y_encoded)

        # 使用模型进行预测并可视化结果
        prediction, proba = predict_and_visualize(clf_best, scaler, label_encoder, sample)

        # 解释预测结果
        shap_values_df = explain_prediction(clf_best, scaler, label_encoder, sample)
        shap_values_df.to_csv('myapp/static/myapp/shap_values.csv', index=False)

        # 将结果传递给模板
        context = {
            'prediction': prediction,
            'proba': proba,
            'shap_values_df': shap_values_df.to_html(),
            'samples': range(1, len(sample) + 1),
            'class_names': label_encoder.classes_,
            'nsv': nsv,
            'al': al,
            'av': av,
            'n': n,
            'λ': λ,
            'sh0': sh0,
        }
        return render(request, 'myapp/predict.html', context)

def probability_result(request):

    if request.method == 'GET':
        # 从查询字符串获取参数
        nsv = float(request.GET.get('nsv'))
        al = float(request.GET.get('al'))
        av = float(request.GET.get('av'))
        n = float(request.GET.get('n'))
        λ = float(request.GET.get('λ'))
        sh0 = float(request.GET.get('sh0'))

        # 创建样本 DataFrame
        sample = pd.DataFrame({
            "nsv": [nsv],
            "al": [al],
            "av": [av],
            "n": [n],
            "λ": [λ],
            "s/h0": [sh0]
        })

        # 加载数据集
        data = pd.read_csv("data/column_墩柱破坏模式.csv")

        # 创建特征列和目标列
        X = data[["nsv", "al", "av", "n", "λ", "s/h0"]]
        y = data["result"]

        # 数据预处理
        X_scaled, y_encoded, label_encoder, scaler = preprocess_data(X, y)

        # 训练模型
        clf_best = train_model(X_scaled, y_encoded)

        # 使用模型进行预测并可视化结果
        prediction, proba = predict_and_visualize(clf_best, scaler, label_encoder, sample)

        # 解释预测结果
        shap_values_df = explain_prediction(clf_best, scaler, label_encoder, sample)
        shap_values_df.to_csv('myapp/static/myapp/shap_values.csv', index=False)

        # 将结果传递给模板
        context = {
            'prediction': prediction,
            'proba': proba,
            'shap_values_df': shap_values_df.to_html(),
            'samples': range(1, len(sample) + 1),
            'class_names': label_encoder.classes_,
        }
        return render(request, 'myapp/probability_result.html', context)

def generate_result(request):
    if request.method == 'GET':
        # 从查询字符串获取参数
        nsv = float(request.GET.get('nsv'))
        al = float(request.GET.get('al'))
        av = float(request.GET.get('av'))
        n = float(request.GET.get('n'))
        λ = float(request.GET.get('λ'))
        sh0 = float(request.GET.get('sh0'))

        # 创建样本 DataFrame
        sample = pd.DataFrame({
            "nsv": [nsv],
            "al": [al],
            "av": [av],
            "n": [n],
            "λ": [λ],
            "s/h0": [sh0]
        })

        # 加载数据集
        data = pd.read_csv("data/column_墩柱破坏模式.csv")

        # 创建特征列和目标列
        X = data[["nsv", "al", "av", "n", "λ", "s/h0"]]
        y = data["result"]

        # 数据预处理
        X_scaled, y_encoded, label_encoder, scaler = preprocess_data(X, y)

        # 训练模型
        clf_best = train_model(X_scaled, y_encoded)

        # 使用模型进行预测并可视化结果
        prediction, proba = predict_and_visualize(clf_best, scaler, label_encoder, sample)

        # 解释预测结果
        shap_values_df = explain_prediction(clf_best, scaler, label_encoder, sample)
        shap_values_df.to_csv('myapp/static/myapp/shap_values.csv', index=False)

        # 将结果传递给模板
        context = {
            'prediction': prediction,
            'proba': proba,
            'shap_values_df': shap_values_df.to_html(),
            'samples': range(1, len(sample) + 1),
            'class_names': label_encoder.classes_,
        }
        return render(request, 'myapp/probability_result.html', context)

def shap_values_result(request):
    if request.method == 'GET':
        # 从查询字符串获取参数
        nsv = float(request.GET.get('nsv'))
        al = float(request.GET.get('al'))
        av = float(request.GET.get('av'))
        n = float(request.GET.get('n'))
        λ = float(request.GET.get('λ'))
        sh0 = float(request.GET.get('sh0'))

        # 创建样本 DataFrame
        sample = pd.DataFrame({
            "nsv": [nsv],
            "al": [al],
            "av": [av],
            "n": [n],
            "λ": [λ],
            "s/h0": [sh0]
        })

        # 加载数据集
        data = pd.read_csv("data/column_墩柱破坏模式.csv")

        # 创建特征列和目标列
        X = data[["nsv", "al", "av", "n", "λ", "s/h0"]]
        y = data["result"]

        # 数据预处理
        X_scaled, y_encoded, label_encoder, scaler = preprocess_data(X, y)

        # 训练模型
        clf_best = train_model(X_scaled, y_encoded)

        # 使用模型进行预测并可视化结果
        prediction, proba = predict_and_visualize(clf_best, scaler, label_encoder, sample)

        # 解释预测结果
        shap_values_df = explain_prediction(clf_best, scaler, label_encoder, sample)
        shap_values_df.to_csv('myapp/static/myapp/shap_values.csv', index=False)

        # 将结果传递给模板
        context = {
            'prediction': prediction,
            'proba': proba,
            'shap_values_df': shap_values_df.to_html(),
            'samples': range(1, len(sample) + 1),
            'class_names': label_encoder.classes_,
        }
        return render(request, 'myapp/shap_values_result.html', context)

def shap_force_plots_result(request):
    if request.method == 'GET':
        # 从查询字符串获取参数
        nsv = float(request.GET.get('nsv'))
        al = float(request.GET.get('al'))
        av = float(request.GET.get('av'))
        n = float(request.GET.get('n'))
        λ = float(request.GET.get('λ'))
        sh0 = float(request.GET.get('sh0'))

        # 创建样本 DataFrame
        sample = pd.DataFrame({
            "nsv": [nsv],
            "al": [al],
            "av": [av],
            "n": [n],
            "λ": [λ],
            "s/h0": [sh0]
        })

        # 加载数据集
        data = pd.read_csv("data/column_墩柱破坏模式.csv")

        # 创建特征列和目标列
        X = data[["nsv", "al", "av", "n", "λ", "s/h0"]]
        y = data["result"]

        # 数据预处理
        X_scaled, y_encoded, label_encoder, scaler = preprocess_data(X, y)

        # 训练模型
        clf_best = train_model(X_scaled, y_encoded)

        # 使用模型进行预测并可视化结果
        prediction, proba = predict_and_visualize(clf_best, scaler, label_encoder, sample)

        # 解释预测结果
        shap_values_df = explain_prediction(clf_best, scaler, label_encoder, sample)
        shap_values_df.to_csv('myapp/static/myapp/shap_values.csv', index=False)

        # 将结果传递给模板
        context = {
            'prediction': prediction,
            'proba': proba,
            'shap_values_df': shap_values_df.to_html(),
            'samples': range(1, len(sample) + 1),
            'class_names': label_encoder.classes_,
        }
        return render(request, 'myapp/shap_force_plots_result.html', context)


def shap_values_data(request):
    if request.method == 'GET':
        # 从查询字符串获取参数
        nsv = float(request.GET.get('nsv'))
        al = float(request.GET.get('al'))
        av = float(request.GET.get('av'))
        n = float(request.GET.get('n'))
        λ = float(request.GET.get('λ'))
        sh0 = float(request.GET.get('sh0'))

        # 创建样本 DataFrame
        sample = pd.DataFrame({
            "nsv": [nsv],
            "al": [al],
            "av": [av],
            "n": [n],
            "λ": [λ],
            "s/h0": [sh0]
        })

        # 加载数据集
        data = pd.read_csv("data/column_墩柱破坏模式.csv")

        # 创建特征列和目标列
        X = data[["nsv", "al", "av", "n", "λ", "s/h0"]]
        y = data["result"]

        # 数据预处理
        X_scaled, y_encoded, label_encoder, scaler = preprocess_data(X, y)

        # 训练模型
        clf_best = train_model(X_scaled, y_encoded)

        # 使用模型进行预测并可视化结果
        prediction, proba = predict_and_visualize(clf_best, scaler, label_encoder, sample)

        # 解释预测结果
        shap_values_df = explain_prediction(clf_best, scaler, label_encoder, sample)
        shap_values_df.to_csv('myapp/static/myapp/shap_values.csv', index=False)

        # 将结果传递给模板
        context = {
            'prediction': prediction,
            'proba': proba,
            'shap_values_df': shap_values_df.to_html(),
            'samples': range(1, len(sample) + 1),
            'class_names': label_encoder.classes_,
        }
        return render(request, 'myapp/shap_values_data.html', context)


