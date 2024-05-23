import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

# 加载数据
data = pd.read_csv('data.csv')

# 筛选数据
player1_data = data[data['选手姓名'] == '张三']
player2_data = data[data['选手姓名'] == '李四']

# 合并数据
data = pd.concat([player1_data, player2_data], ignore_index=True)

#提取目标值（单站积分）
y = data['单站积分']

# 编码类别特征 (使用 get_dummies 简化)
data_encoded = pd.get_dummies(data, columns=['比赛日期', '比赛地点', '比赛级别', '剑种', '持剑手'])

# 标准化数值特征
scaler = StandardScaler()
numeric_cols = ['选手年龄', '选手排名']
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

# 提取特征 (删除选手姓名列)
X = data_encoded.drop(['选手姓名'], axis=1)  # 删除参赛选手姓名列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义模型
models = {
    'RandomForestRegressor': RandomForestRegressor(),
    'SVR': SVR()
}

# 使用 LeaveOneOut 交叉验证
loo = LeaveOneOut()

# 网格搜索优化超参数
best_model = None
best_score = float('inf')  # 初始最佳分数设为无穷大
for name, model in models.items():
    scores = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 在 LeaveOneOut 循环内部定义 param_grid
        if name == 'RandomForestRegressor':
            param_grid = {'n_estimators': [100, 200, 500]}
        elif name == 'SVR':
            param_grid = {'C': [0.1, 1, 10]}

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')  # 使用负均方误差
        grid_search.fit(X_train, y_train)
        scores.append(grid_search.best_score_)

    mean_score = sum(scores) / len(scores)  # 计算平均得分
    if mean_score < best_score:  # 选择均方误差更小的模型
        best_score = mean_score
        best_model = grid_search.best_estimator_

    print(f'{name} - 最佳参数：{grid_search.best_params_}, 平均均方误差：{-mean_score:.4f}')  # 输出均方误差

print(f'\n最终最佳模型：{best_model}')

## 定义新比赛数据 (示例)
player1_new_data = pd.DataFrame({
    '比赛日期': ['2024-06-01'],
    '比赛地点': ['上海'],
    '比赛级别': ['A'],
    '剑种': ['F'],
    '持剑手': ['L'],
    '选手年龄': [15],
    '选手排名': [1]
})

player2_new_data = pd.DataFrame({
    '比赛日期': ['2024-06-01'],
    '比赛地点': ['上海'],
    '比赛级别': ['A'],
    '剑种': ['F'],
    '持剑手': ['R'],
    '选手年龄': [14],
    '选手排名': [10]
})

# 使用最佳模型进行预测 (合并新数据处理)
new_data = pd.concat([player1_new_data, player2_new_data], ignore_index=True)
new_data_encoded = pd.get_dummies(new_data, columns=['比赛日期', '比赛地点', '比赛级别', '剑种', '持剑手'])

# 确保新数据与训练数据具有相同的列
missing_cols = set(X_train.columns) - set(new_data_encoded.columns)
for col in missing_cols:
    new_data_encoded[col] = 0
new_data_encoded = new_data_encoded[X_train.columns]

# 标准化数值特征
new_data_encoded[numeric_cols] = scaler.transform(new_data_encoded[numeric_cols])

# new_data_encoded = new_data_encoded.drop(['选手姓名'], axis=1)

predictions = best_model.predict(new_data_encoded)
print(predictions)