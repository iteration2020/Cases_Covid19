# Импорт необходимых для работы библиотек
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from clickhouse_sqlalchemy import make_session

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
world_cases = []
total_deaths = []
mortality_rate = [] # Показатель смертности
for i in range(num_dates):
    confirmed_sum = Confirmed[ck[i]].sum()
    death_sum = Deaths[dk[i]].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum / confirmed_sum)

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
        st.write("Вы выбрали Пункт 3")
    elif choice == "О проекте":
        st.write("Вы выбрали Пункт 4")



# Обучение модели методом Полиноминальной регрессии
def learing():



# Дневной прирост
def daily_increase():
    d = []
    for i in range(len()):
        if i == 0:
            d.append([0])
        else:
            d.append([i] - [i - 1])
    return d

# Среднее изменение
def moving_average(data, window_size):
    moving_average = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_average.append(np.mean(data[i:i + window_size]))
        else:
            moving_average.append(np.mean(data[i:len(data)]))
    return moving_average