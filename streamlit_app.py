import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# --------------------------
# 1. Настройка и Загрузка Данных
# --------------------------

st.set_page_config(
    page_title="Прогнозирование Зарплаты (ML-Baseline)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 Модель Прогнозирования Зарплаты")
st.subheader("Базовая (Линейная Регрессия) и Улучшенная (Случайный Лес) Модели")

# Кэширование данных и модели для ускорения работы Streamlit
@st.cache_data
def load_data():
    """Загрузка данных и обработка пропущенных значений."""
    
    data_path = 'Salary_Data.csv'
    
    try:
        # Проверяем наличие файла, прежде чем читать
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл '{data_path}' не найден.")
            
        # Пытаемся загрузить реальный файл данных
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        # Если файл не найден, выводим ошибку в Streamlit и используем фиктивные данные
        st.error(f"Файл '{data_path}' не найден в директории приложения! Используются фиктивные данные для демонстрации.")
        st.warning("Пожалуйста, убедитесь, что файл 'Salary_Data.csv' был загружен в то же место, что и скрипт 'streamlit_app.py'.")
        
        # Фиктивные данные для демонстрации
        data = pd.DataFrame({
            'Age': [25, 28, 30, 32, 35, 38, 40, 42, 45, 50],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'Education Level': ["Bachelor's", "Master's", "Bachelor's", "PhD", "Master's", "Bachelor's", "PhD", "Master's", "Bachelor's", "PhD"],
            'Job Title': ['Software Engineer', 'Data Scientist', 'Manager', 'Analyst', 'Software Engineer', 'Data Scientist', 'Manager', 'Analyst', 'Software Engineer', 'Data Scientist'],
            'Years of Experience': [1.5, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0],
            'Salary': [50000, 70000, 90000, 110000, 130000, 140000, 160000, 175000, 190000, 220000]
        })
        
    # Удаление пропущенных значений
    data.dropna(inplace=True)
    return data

data = load_data()

# 2. Подготовка Препроцессора 
numerical_features = ['Age', 'Years of Experience']
categorical_features = ['Gender', 'Education Level', 'Job Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)

# 3. Обучение Моделей
@st.cache_resource
def train_models(data, preprocessor):
    """Обучает обе модели (Линейная Регрессия и Случайный Лес)."""
    
    X = data.drop('Salary', axis=1)
    # Применяем конвертацию, как запрошено пользователем (деление на 10 для сомони)
    y = data['Salary'] / 10 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3.1. Линейная Регрессия (Baseline) ---
    linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', LinearRegression())])
    linear_pipeline.fit(X_train, y_train)
    y_pred_linear = linear_pipeline.predict(X_test)
    rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
    r2_linear = r2_score(y_test, y_pred_linear)
    
    # --- 3.2. Случайный Лес ---
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    # Формируем и возвращаем результаты, чтобы их можно было использовать в Streamlit
    results = pd.DataFrame({
        'Модель': ['Линейная Регрессия (Baseline)', 'Случайный Лес (Улучшенная)'],
        'R²': [r2_linear, r2_rf],
        'RMSE': [rmse_linear, rmse_rf]
    })

    return linear_pipeline, rf_pipeline, results

linear_pipeline, rf_pipeline, results = train_models(data, preprocessor)

# --------------------------
# 4. Секция Анализа Данных
# --------------------------

st.header("📊 1. Обзор и Визуализация Данных")

if st.checkbox('Показать исходные данные и статистику', value=True):
    st.subheader("Исходный Датасет")
    st.dataframe(data.head())
    st.subheader("Сводная Статистика")
    st.dataframe(data.describe())

st.markdown("---")

st.subheader("Визуализация ключевых распределений")

# Настройка стиля для графиков
sns.set_style("whitegrid")

# Гистограммы для численных признаков
fig_hist, axes_hist = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(data['Age'], kde=True, ax=axes_hist[0])
axes_hist[0].set_title('Распределение Возраста')
axes_hist[0].set_xlabel('Возраст')

sns.histplot(data['Years of Experience'], kde=True, ax=axes_hist[1])
axes_hist[1].set_title('Распределение Стажа Работы')
axes_hist[1].set_xlabel('Стаж работы (годы)')

sns.histplot(data['Salary'], kde=True, ax=axes_hist[2])
axes_hist[2].set_title('Распределение Зарплаты')
axes_hist[2].set_xlabel('Зарплата')

st.pyplot(fig_hist)


# Диаграммы рассеяния
fig_scatter, axes_scatter = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(x='Years of Experience', y='Salary', data=data, hue='Education Level', size='Age', sizes=(20, 400), alpha=0.7, ax=axes_scatter[0])
axes_scatter[0].set_title('Зарплата vs. Стаж Работы')
axes_scatter[0].set_xlabel('Стаж работы (годы)')
axes_scatter[0].set_ylabel('Зарплата')

sns.scatterplot(x='Age', y='Salary', data=data, hue='Gender', size='Years of Experience', sizes=(20, 400), alpha=0.7, ax=axes_scatter[1])
axes_scatter[1].set_title('Зарплата vs. Возраст')
axes_scatter[1].set_xlabel('Возраст')
axes_scatter[1].set_ylabel('Зарплата')

st.pyplot(fig_scatter)


# Графики распределения для категориальных признаков
fig_cat, axes_cat = plt.subplots(1, 2, figsize=(16, 5))

sns.countplot(y='Education Level', data=data, order=data['Education Level'].value_counts().index, palette='viridis', ax=axes_cat[0])
axes_cat[0].set_title('Распределение по Уровню Образования')
axes_cat[0].set_xlabel('Количество')

sns.countplot(x='Gender', data=data, palette='coolwarm', ax=axes_cat[1])
axes_cat[1].set_title('Распределение по Полу')
axes_cat[1].set_xlabel('Пол')

st.pyplot(fig_cat)

# Barplot для Job Title
if data['Job Title'].nunique() < 30:
    fig_job, ax_job = plt.subplots(figsize=(10, 6))
    sns.barplot(y='Job Title', x='Salary', data=data.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).reset_index(), palette='magma', ax=ax_job)
    ax_job.set_title('Средняя Зарплата по Должности')
    ax_job.set_xlabel('Средняя Зарплата')
    ax_job.set_ylabel('Должность')
    st.pyplot(fig_job)
else:
    st.info(f"В датасете {data['Job Title'].nunique()} уникальных должностей. График средней зарплаты по должности не строится из-за большого числа категорий.")

# --------------------------
# 5. Секция Оценки Модели
# --------------------------

st.header("⚙️ 2. Оценка Моделей")

st.markdown("Сравнение базовой модели (Линейная Регрессия) и улучшенной модели (Случайный Лес) на тестовом наборе данных. Единица измерения ошибки: **сомонӣ**.")

col1, col2 = st.columns(2)

# Используем результаты из DataFrame 'results', а не глобальные переменные
with col1:
    st.subheader("Линейная Регрессия (Baseline)")
    st.metric(
        label="Корень из среднеквадратичной ошибки (RMSE)",
        value=f"{results.loc[0, 'RMSE']:,.2f} сомонӣ"
    )
    st.metric(
        label="Коэффициент детерминации (R²)",
        value=f"{results.loc[0, 'R²']:.4f}"
    )

with col2:
    st.subheader("Случайный Лес (Улучшенная)")
    st.metric(
        label="Корень из среднеквадратичной ошибки (RMSE)",
        value=f"{results.loc[1, 'RMSE']:,.2f} сомонӣ"
    )
    st.metric(
        label="Коэффициент детерминации (R²)",
        value=f"{results.loc[1, 'R²']:.4f}"
    )

st.dataframe(results, hide_index=True)


# --------------------------
# 6. Секция Прогнозирования
# --------------------------

st.header("🔮 3. Прогнозирование Зарплаты")
st.markdown("Введите данные сотрудника, чтобы получить прогноз зарплаты.")

# Выбор модели
model_selection = st.selectbox(
    'Выберите модель для прогнозирования:',
    options=['Случайный Лес (Улучшенная)', 'Линейная Регрессия (Baseline)'],
    index=0 # По умолчанию выбираем лучшую
)

if model_selection == 'Линейная Регрессия (Baseline)':
    model = linear_pipeline
    st.info("Используется **Линейная Регрессия**.")
else:
    model = rf_pipeline
    st.info("Используется **Случайный Лес**.")


# Поля для ввода данных
input_cols = st.columns(3)

with input_cols[0]:
    age = st.slider("Возраст (Age)", min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=35, step=1)
    gender = st.selectbox("Пол (Gender)", options=data['Gender'].unique())

with input_cols[1]:
    experience = st.slider("Стаж работы (Years of Experience)", min_value=0.0, max_value=float(data['Years of Experience'].max()), value=7.0, step=0.5)
    education = st.selectbox("Уровень Образования (Education Level)", options=data['Education Level'].unique())

with input_cols[2]:
    # Убедимся, что Job Title не слишком длинный для отображения
    job_options = sorted(data['Job Title'].unique())
    job_title = st.selectbox("Должность (Job Title)", options=job_options)


# Формирование входного DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education],
    'Job Title': [job_title],
    'Years of Experience': [experience]
})

# Кнопка для предсказания
if st.button('Сделать Прогноз Зарплаты', type="primary"):
    try:
        # Прогноз
        predicted_salary = model.predict(input_data)[0]
        
        # Вывод результата
        st.success(f"**Прогноз Зарплаты:**")
        st.balloons()
        
        # Форматирование вывода
        st.markdown(f"## {predicted_salary:,.2f} сомонӣ")
        st.caption("Прогноз сделан на основе выбранной вами модели.")

    except Exception as e:
        st.error(f"Произошла ошибка при прогнозировании. Пожалуйста, проверьте входные данные. Детали: {e}")
