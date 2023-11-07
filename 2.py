import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    Найти и выгрузить данные. Вывести, провести предобработку и описать признаки.
    """

    df = pd.read_csv(Path('vgsales.csv'))

    print('Количество пропусков:')
    print(df.isna().sum())

    df.dropna(inplace=True)

    print('Количество пропусков после очистки:')
    print(df.isna().sum())

    print('Информация о столбцах:')
    print(df.info())

    numeric_columns = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

    """
    Построить корреляционную матрицу по одной целевой переменной. Определить наиболее коррелирующую переменную, продолжить с ней работу в следующем пункте.
    """

    correlation_matrix = df[numeric_columns].corr()
    print(correlation_matrix)

    # Наибольшей корреляцией обладают столбцы NA_Sales (Продажи в Северной Америке) и Global_Sales (Мировые продажи)

    """
    Реализовать регрессию вручную, отобразить наклон, сдвиг и MSE.
    """

    model = LinearRegression()

    X = np.array(df[['NA_Sales']], type(float))
    y = np.array(df['Global_Sales'], type(float))

    model.fit(X, y)

    model_coef = model.coef_[0]
    model_intercept = model.intercept_

    linear_regression_line = model_coef * X + model_intercept

    mse = mean_squared_error(linear_regression_line, y)

    print(f'\nУгловой коэффициент регрессии: {model_coef}')
    print(f'Сдвиг: {model_intercept}')
    print(f'MSE: {mse}')

    # Угловой коэффициент регрессии равен 1.794, сдвиг равен 0.064, а среднеквадртаичная ошибка 0.28

    """
    Визуализировать регрессию на графике.
    """

    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.scatter(X, y, alpha=0.3)

    plt.plot(X, model.predict(X), color='green', linewidth=3)
    plt.xlabel('Продажи в СА (млн. копий)')
    plt.ylabel('Мировые продажи (млн. копий)')
    plt.show()
