import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    street = np.array([80, 98, 75, 91, 78])
    garage = np.array([100, 82, 105, 89, 102])

    correlation = np.corrcoef(street, garage)[0, 1]
    print(f'Кореляция равна {correlation}')

    plt.grid(True)
    plt.title('Диаграмма рассеяния', fontsize=20)
    plt.xlabel("Улица")
    plt.ylabel("Парковка")
    plt.scatter(x=street, y=garage, marker='o', color='green')
    plt.show()

"""
Вывод: коэффициент корреляции практически равен -1, следовательно присутствует сильная отрицательная корреляция переменных
"""