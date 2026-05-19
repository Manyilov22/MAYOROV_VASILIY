import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from scipy.stats import kurtosis


# ЛОГИРОВАНИЕ

logging.basicConfig(
    filename="data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

logging.info("Программа запущена")

# ПОЛУЧЕНИЕ ДАННЫХ ЧЕРЕЗ API

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 55.75,
    "longitude": 37.61,
    "hourly": "temperature_2m",
    "past_days": 60,
    "forecast_days": 1
}

response = requests.get(url, params=params)

data = response.json()

logging.info("Данные успешно получены через API")

# СОЗДАНИЕ DATAFRAME

df = pd.DataFrame({
    "time": data["hourly"]["time"],
    "temperature": data["hourly"]["temperature_2m"]
})

# Преобразование времени

df["time"] = pd.to_datetime(df["time"])

logging.info("DataFrame успешно создан")

# ПРОВЕРКА ПРОПУСКОВ

missing_values = df["temperature"].isna().sum()

print("Количество пропусков:", missing_values)

logging.info(f"Количество пропусков: {missing_values}")

# ЗАПОЛНЕНИЕ ПРОПУСКОВ

median_value = df["temperature"].median()

df["temperature"] = df["temperature"].fillna(median_value)

logging.info("Пропуски заполнены медианным значением")

# ОЧИСТКА ОТ ВЫБРОСОВ

q1 = df["temperature"].quantile(0.25)

q3 = df["temperature"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr

upper_bound = q3 + 1.5 * iqr

before_rows = len(df)

df = df[
    (df["temperature"] >= lower_bound) &
    (df["temperature"] <= upper_bound)
]

after_rows = len(df)

deleted_rows = before_rows - after_rows

print("Удалено выбросов:", deleted_rows)

logging.info(f"Удалено выбросов: {deleted_rows}")

# ФОРМИРОВАНИЕ ОБЪЕКТА SERIES

temperature_series = df["temperature"]

temperature_array = np.array(temperature_series)

logging.info("Series и NumPy массив успешно созданы")

# РАСЧЕТ ХАРАКТЕРИСТИК
# ВАРИАНТ 19

# Сумма значений

sum_value = df["temperature"].sum()

# Среднее арифметическое

mean_value = df["temperature"].mean()

# Медиана

median_value = df["temperature"].median()

# Интерквартильный размах

iqr_value = iqr

# Эксцесс

kurtosis_value = kurtosis(df["temperature"])

# ВЫВОД ХАРАКТЕРИСТИК

print("\nСТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ")

print("Сумма значений:", sum_value)

print("Среднее арифметическое:", mean_value)

print("Медиана:", median_value)

print("Интерквартильный размах:", iqr_value)

print("Эксцесс:", kurtosis_value)

logging.info("Статистические характеристики рассчитаны")

# ГРУППИРОВКА ПО ДНЯМ НЕДЕЛИ

df["day_of_week"] = df["time"].dt.day_name()

grouped_data = df.groupby("day_of_week")["temperature"].mean()

print("\nСРЕДНЯЯ ТЕМПЕРАТУРА ПО ДНЯМ НЕДЕЛИ")

print(grouped_data)

logging.info("Группировка по дням недели выполнена")

# ТРИ НАИБОЛЬШИХ ЗНАЧЕНИЯ

top_3_max = df.nlargest(3, "temperature")

print("\nТРИ НАИБОЛЬШИХ ЗНАЧЕНИЯ")

print(top_3_max[["time", "temperature"]])

# ТРИ НАИМЕНЬШИХ ЗНАЧЕНИЯ

top_3_min = df.nsmallest(3, "temperature")

print("\nТРИ НАИМЕНЬШИХ ЗНАЧЕНИЯ")

print(top_3_min[["time", "temperature"]])

logging.info("Максимальные и минимальные значения найдены")

# ПРОВЕРКА ГИПОТЕЗЫ

print("\nГИПОТЕЗА:")

print("Во временном ряду присутствуют колебания температуры.")

logging.info("Гипотеза сформулирована")

# ВИЗУАЛИЗАЦИЯ

# Линейный график

plt.figure(figsize=(14, 6))

plt.plot(df["time"], df["temperature"])

plt.title("Изменение температуры во времени")

plt.xlabel("Дата")

plt.ylabel("Температура")

plt.grid()

plt.show()

# Гистограмма

plt.figure(figsize=(10, 6))

plt.hist(df["temperature"], bins=30)

plt.title("Гистограмма распределения температуры")

plt.xlabel("Температура")

plt.ylabel("Количество значений")

plt.show()

# Boxplot

plt.figure(figsize=(8, 5))

sns.boxplot(x=df["temperature"])

plt.title("Boxplot температуры")

plt.show()

logging.info("Визуализация построена")

print("\nПрограмма успешно завершена")
