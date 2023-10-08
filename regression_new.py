#Импорт необходимых для работы библиотек
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Импорт датасетов
# Численность с подтвержденным диагнозом COVID19
Confirmed_DF = pd.read_csv('/DS/COVID19_confirmed_global.csv')
# Чиcленность умерших от COVID19
Deaths_DF = pd.read_csv('/DS/covid19_deaths_global.csv')
# Последние данные по заболеваемости
Latest_DF = pd.read_csv('/DS/covid_19_daily_reports_08-25-2022_latest.csv')
# Медицинские данные по США
US_med_DF = pd.read_csv('/DS/covid_19_daily_reports_us_08-25-2022.csv')