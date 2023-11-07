def task0():
    """
    Загрузить данные: 'insurance.csv'. Вывести и провести предобработку. Вывести список уникальных регионов.
    """
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(Path('insurance.csv'))

    print('Количество пропусков:')
    print(df.isna().sum())
    print('\nУникальные регионы:')
    print(df['region'].unique())

def task1():
    """
    Выполнить однофакторный ANOVA тест, чтобы проверить влияние региона на индекс массы тела (BMI), используя первый способ, через библиотеку Scipy
    """
    import pandas as pd
    from scipy.stats import f_oneway
    from pathlib import Path

    df = pd.read_csv(Path('insurance.csv'))

    southwest = df[df['region'] == 'southwest']['bmi']
    southeast = df[df['region'] == 'southeast']['bmi']
    northwest = df[df['region'] == 'northwest']['bmi']
    northeast = df[df['region'] == 'northeast']['bmi']

    print(f_oneway(southwest, southeast, northwest, northeast))
    """
    Вывод: p-значение сильно меньше 0.05, следовательно регион оказывает статистически значимое влияние на индекс массы тела
    """

def task2():
    """
    Выполнить однофакторный ANOVA тест, чтобы проверить влияние региона на индекс массы тела (BMI), используя второй способ, с помощью функции anova_lm() из библиотеки statsmodels.
    """
    import pandas as pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from pathlib import Path

    df = pd.read_csv(Path('insurance.csv'))

    model = ols('bmi ~ region', data=df).fit()
    print(anova_lm(model, typ=1))
    """
    Вывод: p-значение сильно меньше 0.05, следовательно регион оказывает статистически значимое влияние на индекс массы тела
    """


def task3():
    """
    С помощью t критерия Стьюдента перебрать все пары. Определить поправку Бонферрони. Сделать выводы.
    """
    import pandas as pd
    from itertools import combinations
    from scipy.stats import ttest_ind
    from pathlib import Path

    df = pd.read_csv(Path('insurance.csv'))

    regions = df['region'].unique()

    bmis = {region: df[df['region'] == region]['bmi'] for region in regions}

    region_pairs = combinations(regions, 2)

    result = pd.DataFrame(columns=['Регион 1', 'Регион 2', 'p-значение'])

    for region1, region2 in region_pairs:
        new_row = {'Регион 1': region1,
                   'Регион 2': region2,
                   'p-значение': ttest_ind(bmis[region1], bmis[region2]).pvalue}
        result.loc[len(result)] = new_row

    bonferroni_correction = len(regions)
    result['скорректированное p-значение'] = result['p-значение'] * bonferroni_correction

    result['скорректированное p-значение'] = result['скорректированное p-значение'].round(5)

    print(f'Корректировка Бонферрони равна {bonferroni_correction}\n')
    print(result)
    """
    Вывод: критерий Бонферрони равен 4, так как всего общая совокупность была разделена на группы по 4 регионам. 
    Скорректированное p-значеие значительно меньше 0.05 в трех парах: southwest-southeast, southeast-northwest, southeast-northeast, что приводит к выводу о том. что юго-восточный (southeeast) регион имеет наиболее значительное отличие от остальных
    Также это значение меньше 0.05 в парах southwest-northwest и shouthwest-northwest, что показывает то, что группа soutchwest также отличается от остальных
    """

def task4():
    """
    Выполнить пост-хок тесты Тьюки и построить график.
    """
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    df = pd.read_csv(Path('insurance.csv'))

    tukey = pairwise_tukeyhsd(endog=df['bmi'], groups=df['region'], alpha=0.05)
    tukey.plot_simultaneous()
    plt.vlines(x=df['bmi'].mean(), ymin=-1, ymax=4, color='red')
    print(tukey.summary())
    plt.show()
    """
    Вывод: По графику видно, что области northeast и northwest перекрываются, поэтому их различия не существенные, а две другие группы отличаются от них, причем группа southeast имеет сильное отличие
    """

def task5():
    """
    Выполнить двухфакторный ANOVA тест, чтобы проверить влияние региона и пола на индекс массы тела (BMI), используя функцию anova_lm() из библиотеки statsmodels.
    """
    import pandas as pd
    from pathlib import Path
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df = pd.read_csv(Path('insurance.csv'))

    model = ols('bmi ~ C(region) + C(sex) + C(region):C(sex)', data=df).fit()
    print(anova_lm(model, typ=2))
    """
    Вывод: p-знаечние для региона сильно меньше 0.05, следовательно этот фактор оказывает статистически значимое влияние на индекс массы тела. p-значение для парамтера "пол" значительно выше 0.05. что означает. что оно не оказывает влияния на ИМТ. Наконец, p-значение для регион*пол составляет более 0.165, а значит между этими параметрами нет статистически важного влияния
    """

def task6():
    """
    Выполнить пост-хок тесты Тьюки и построить график.
    """
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    df = pd.read_csv(Path('insurance.csv'))

    df['grouped'] = df['region'] + '/' + df['sex']
    tukey = pairwise_tukeyhsd(endog=df['bmi'], groups=df['grouped'], alpha=0.05)
    tukey.plot_simultaneous()
    print(tukey.summary())
    plt.show()


if __name__ == '__main__':
    #task0()
    #task1()
    #task2()
    #task3()
    #task4()
    #task5()
    task6()
    pass