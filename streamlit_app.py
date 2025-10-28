import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# 1. Загрузка данных
data = pd.read_csv('Salary_Data.csv')



data.isna().sum()
data
# 1.1. Визуализация данных
print("\n--- Визуализация данных ---")

# Настройка стиля для графиков
sns.set_style("whitegrid")

# Гистограммы для численных признаков
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(data['Age'], kde=True)
plt.title('Распределение Возраста')
plt.xlabel('Возраст')
plt.ylabel('Частота')

plt.subplot(1, 3, 2)
sns.histplot(data['Years of Experience'], kde=True)
plt.title('Распределение Стажа Работы')
plt.xlabel('Стаж работы (годы)')
plt.ylabel('Частота')

plt.subplot(1, 3, 3)
sns.histplot(data['Salary'], kde=True)
plt.title('Распределение Зарплаты')
plt.xlabel('Зарплата')
plt.ylabel('Частота')
plt.tight_layout()
plt.show()

# Диаграммы рассеяния для 'Salary' против численных признаков
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x='Years of Experience', y='Salary', data=data, hue='Education Level', size='Age', sizes=(20, 400), alpha=0.7)
plt.title('Зарплата vs. Стаж Работы')
plt.xlabel('Стаж работы (годы)')
plt.ylabel('Зарплата')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Age', y='Salary', data=data, hue='Gender', size='Years of Experience', sizes=(20, 400), alpha=0.7)
plt.title('Зарплата vs. Возраст')
plt.xlabel('Возраст')
plt.ylabel('Зарплата')
plt.tight_layout()
plt.show()

# Графики распределения для категориальных признаков
# Для Education Level и Gender (если Job Title имеет слишком много уникальных значений, его лучше не рисовать)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.countplot(y='Education Level', data=data, order=data['Education Level'].value_counts().index, palette='viridis')
plt.title('Распределение по Уровню Образования')
plt.xlabel('Количество')
plt.ylabel('Уровень Образования')

plt.subplot(1, 2, 2)
sns.countplot(x='Gender', data=data, palette='coolwarm')
plt.title('Распределение по Полу')
plt.xlabel('Пол')
plt.ylabel('Количество')
plt.tight_layout()
plt.show()

# Если Job Title имеет не слишком много уникальных значений (например, до 20-30), можно визуализировать
if data['Job Title'].nunique() < 30:
    plt.figure(figsize=(12, 7))
    sns.barplot(y='Job Title', x='Salary', data=data.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).reset_index(), palette='magma')
    plt.title('Средняя Зарплата по Должности')
    plt.xlabel('Средняя Зарплата')
    plt.ylabel('Должность')
    plt.tight_layout()
    plt.show()
else:
    print("\n'Job Title' имеет слишком много уникальных значений для эффективной визуализации.")
data.dropna(inplace = True)
# 2. Определение признаков (X) и целевой переменной (y)
X = data.drop('Salary', axis=1)
y = data['Salary']/10

# 3. Разделение признаков на типы
# Численные признаки (будут стандартизированы)
numerical_features = ['Age', 'Years of Experience']
# Категориальные признаки (будут закодированы)
categorical_features = ['Gender', 'Education Level', 'Job Title']

# 4. Создание препроцессора (Pipeline для преобразования признаков)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' # Отбрасываем любые другие столбцы, если есть
)

# 5. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Создание и обучение конвейера (Pipeline) для Линейной Регрессии (Baseline)
linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', LinearRegression())])

linear_pipeline.fit(X_train, y_train)

# 7. Предсказание и оценка
y_pred_linear = linear_pipeline.predict(X_test)

rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print("\n--- Оценка Модели: Линейная Регрессия (Baseline) ---")
print(f"Корень из среднеквадратичной ошибки (RMSE):  {rmse_rf:,.2f} сомонӣ")
print(f"Коэффициент детерминации (R²): {r2_linear:.4f}")
y
# 1. Создание и обучение конвейера для Случайного Леса
# Для Случайного Леса стандартизация численных признаков не обязательна, но не повредит,
# поэтому мы используем тот же препроцессор.
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

rf_pipeline.fit(X_train, y_train)

# 2. Предсказание и оценка
y_pred_rf = rf_pipeline.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Оценка Модели: Случайный Лес (Улучшенная) ---")
print(f"Корень из среднеквадратичной ошибки (RMSE): {rmse_rf:,.2f} сомонӣ")
print(f"Коэффициент детерминации (R²): {r2_rf:.4f}")

# 3. Сводная таблица результатов
print("\n--- Сравнение Моделей по R² ---")
results = pd.DataFrame({
    'Модель': ['Линейная Регрессия (Baseline)', 'Случайный Лес (Улучшенная)'],
    'R²': [r2_linear, r2_rf],
    'RMSE': [rmse_linear, rmse_rf]
})
print(results.to_markdown(index=False, floatfmt=".4f"))
