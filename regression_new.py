# Импорт необходимых для работы библиотек
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from clickhouse_sqlalchemy import make_session
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge

# Подключение БД (ClickHouse)
engine = create_engine('clickhouse://default:@localhost:9000/COVID')
session = make_session(engine)



# Импорт датасетов из базы ClickHouse
# Численность с подтвержденным диагнозом COVID19
query = 'SELECT * FROM confirmed_global'
Confirmed_DF = pd.read_sql(query,engine)
# Чиcленность умерших от COVID19
query = 'SELECT * FROM deaths_global'
Deaths_DF = pd.read_sql(query,engine)
# Последние данные по заболеваемости
query = 'SELECT * FROM reports_latest'
Latest_DF= pd.read_sql(query,engine)
# Медицинские данные по США
query = 'SELECT * FROM reports_us'
US_med_DF = pd.read_sql(query,engine)

# Предобработка
confirmed_cols = Confirmed_DF.keys()
deaths_cols = Deaths_DF.keys()
Confirmed = Confirmed_DF.loc[:, confirmed_cols[4]:]
Deaths = Deaths_DF.loc[:,deaths_cols[4]:]
num_dates = len(Confirmed.keys())
ck = Confirmed.keys()
dk = Deaths.keys()
world_cases =
total_deaths =

def daily_increase(data):
    d = []
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d

def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i+window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average

window = 7

# confirmed cases
world_daily_increase = daily_increase(world_cases)
world_confirmed_avg = moving_average(world_cases, window)
world_daily_increase_avg = moving_average(world_daily_increase, window)
# deaths
world_daily_death = daily_increase(total_deaths)
world_death_avg = moving_average(total_deaths, window)
world_daily_death_avg = moving_average(world_daily_death, window)

# Меню веб-клиента
def mainmenu():
    st.sidebar.title("Меню")

    menu_options = ["Исходные данные", "Визуалиация и аналитика", "Обучение модели", "О проекте"]
    choice = st.sidebar.selectbox("Выберите пункт меню", menu_options)
    if choice == "Исходные данные":
        view = st.sidebar.selectbox('Выберите таблицу',['confirmed_global','death_global','report_latest','report_us'])
        if view == 'confirmed_global':
            st.write(Confirmed_DF)
        elif view == 'death_global':
            st.write(Deaths_DF)
        elif view == 'reports_latest':
            st.write(Latest_DF)
        elif view == 'report_us':
            st.write(US_med_DF)
    elif choice == "Визуалиация и аналитика":
        st.write("Вы выбрали Пункт 2")
    elif choice == "Обучение модели":
        days_to_skip = st.number_input("Количество дней для пропуска?")
        learning()
    elif choice == "О проекте":
        st.write("Вы выбрали Пункт 4")

# Обучение модели методом Полиноминальной регрессии
def learning():
    for i in range(num_dates):
        confirmed_sum = Confirmed[ck[i]].sum()
        death_sum = Deaths[dk[i]].sum()
        world_cases.append(confirmed_sum)
        total_deaths.append(death_sum)
        mortality_rate.append(death_sum / confirmed_sum)
    days_since_1_22 = np.array([i for i in range(len(ck))]).reshape(-1, 1)
    world_cases = np.array(world_cases).reshape(-1, 1)
    total_deaths = np.array(total_deaths).reshape(-1, 1)
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(
        days_since_1_22[days_to_skip:], world_cases[days_to_skip:], test_size=0.07, shuffle=False)
    days_in_future = 10
    future_forcast = np.array([i for i in range(len(ck) + days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forcast[:-10]
    poly = PolynomialFeatures(degree=3)
    poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
    poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
    poly_future_forcast = poly.fit_transform(future_forcast)
    linear_model = LinearRegression(fit_intercept=False)
    linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
    test_linear_pred = linear_model.predict(poly_X_test_confirmed)
    linear_pred = linear_model.predict(poly_future_forcast)
    print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
    print('MSE:', mean_squared_error(test_linear_pred, y_test_confirmed))
    print(linear_model.coef_)
    plt.plot(y_test_confirmed)
    plt.plot(test_linear_pred)
    plt.legend(['Test Data', 'Polynomial Regression Predictions'])
    st.pyplot()
    mae = mean_absolute_error(test_linear_pred, y_test_confirmed)
    mse = mean_squared_error(test_linear_pred, y_test_confirmed)
    st.write('MAE:', mae)
    st.write('MSE:', mse)
