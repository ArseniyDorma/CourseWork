import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import re

warnings.filterwarnings('ignore')

__version__ = "1.0.0"
__author__ = "Data Analysis System"
__description__ = "Система первичного анализа табличных данных"

"""
===============================================================================
СПЕЦИФИКАЦИЯ СИСТЕМЫ ПЕРВИЧНОГО АНАЛИЗА ДАННЫХ v1.0
===============================================================================

НАЗНАЧЕНИЕ:
    Выполнение первичного анализа табличных данных:
    1. Загрузка и визуализация данных
    2. Статистический анализ
    3. Выявление специфических особенностей

ОСНОВНЫЕ КОМПОНЕНТЫ:
    1. CustomCSVParser - парсинг CSV файлов (стандартный формат или данные в одном столбце)
    2. DataAnalyzer - основной класс для анализа данных

ВХОДНЫЕ ДАННЫЕ:
    CSV файл с табличными данными

ВЫХОДНЫЕ ДАННЫЕ:
    - Визуализации (графики распределений, корреляций)
    - Статистические метрики
    - Выявленные особенности (выбросы, дисбаланс, пропуски)

ЗАПУСК:
    python analyze.py

КОМАНДЫ:
    exit - выход из программы
    spec - показать спецификацию
===============================================================================
"""


# ============================================
# 1. ПАРСЕР CSV ФАЙЛОВ
# ============================================

class CustomCSVParser:
    """Парсер для CSV файлов с данными в одном столбце"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def parse(self):
        """Загрузка и парсинг CSV файла"""
        try:
            df = pd.read_csv(self.file_path, header=None)

            if len(df.columns) == 1:
                print("Обнаружен формат: данные в одном столбце")
                return self.parse_single_column(df)
            else:
                print("Обнаружен стандартный CSV формат")
                # Преобразуем названия столбцов в строки
                df.columns = [str(col) for col in df.columns]
                return df

        except Exception as e:
            print(f"Ошибка при загрузке: {e}")
            return None

    def parse_single_column(self, df):
        """Парсинг данных из одного столбца"""
        data_list = df[0].tolist()

        # Пытаемся определить структуру данных
        parsed_rows = []

        i = 0
        while i < len(data_list):
            item = str(data_list[i]) if pd.notna(data_list[i]) else ''

            # Ищем индекс (цифра в начале)
            if re.match(r'^\d+$', item.strip()):
                idx = item.strip()
                i += 1

                # Извлекаем вопрос
                if i < len(data_list):
                    question = str(data_list[i]) if pd.notna(data_list[i]) else ''
                    question = question.strip('"').strip()
                    i += 1

                    # Извлекаем SQL запрос
                    if i < len(data_list):
                        sql = str(data_list[i]) if pd.notna(data_list[i]) else ''
                        sql = sql.strip('"').strip()
                        i += 1

                        # Извлекаем схему (может занимать несколько строк)
                        schema_lines = []
                        while i < len(data_list) and not re.match(r'^\d+$', str(data_list[i]).strip()):
                            schema_line = str(data_list[i]) if pd.notna(data_list[i]) else ''
                            if schema_line.strip() and schema_line.strip() not in ['', ',', 'nan']:
                                schema_lines.append(schema_line.strip('"').strip())
                            i += 1

                        schema = ' '.join(schema_lines)

                        parsed_rows.append({
                            'index': idx,
                            'question': question,
                            'sql_query': sql,
                            'table_schema': schema
                        })
            else:
                i += 1

        if parsed_rows:
            result_df = pd.DataFrame(parsed_rows)
            # Преобразуем названия столбцов в строки
            result_df.columns = [str(col) for col in result_df.columns]
            return result_df
        else:
            # Если не удалось распарсить как текст, возвращаем исходные данные
            df.columns = [str(col) for col in df.columns]
            return df


# ============================================
# 2. АНАЛИЗАТОР ДАННЫХ
# ============================================

class DataAnalyzer:
    """Класс для первичного анализа табличных данных"""

    def __init__(self, data=None):
        self.data = data
        self.numeric_cols = []
        self.categorical_cols = []
        self.date_cols = []

    def load_data(self, file_path):
        """Загрузка данных из файла"""
        parser = CustomCSVParser(file_path)
        self.data = parser.parse()

        if self.data is None:
            print("Ошибка: не удалось загрузить данные")
            return False

        # Приводим все названия столбцов к строковому типу
        self.data.columns = [str(col) for col in self.data.columns]

        print(f"\nДанные успешно загружены")
        print(f"Размер данных: {self.data.shape[0]} строк x {self.data.shape[1]} столбцов")

        # Безопасное отображение названий столбцов
        col_names = [str(col) for col in self.data.columns.tolist()]
        print(f"\nСтолбцы: {', '.join(col_names[:10])}")
        if len(col_names) > 10:
            print(f"... и еще {len(col_names) - 10} столбцов")

        return True

    def identify_column_types(self):
        """Определение типов столбцов"""
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Определяем столбцы с датами
        self.date_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    pd.to_datetime(self.data[col])
                    self.date_cols.append(col)
                except:
                    pass

        print(f"\nТипы столбцов:")
        print(f"  Числовые ({len(self.numeric_cols)}): {', '.join([str(c) for c in self.numeric_cols[:10]])}")
        print(
            f"  Категориальные ({len(self.categorical_cols)}): {', '.join([str(c) for c in self.categorical_cols[:10]])}")
        print(f"  Дата/время ({len(self.date_cols)}): {', '.join([str(c) for c in self.date_cols[:10]])}")

    def visualize_data(self):
        """Визуализация данных"""
        if self.data is None:
            print("Ошибка: данные не загружены")
            return

        self.identify_column_types()

        # Проверяем, есть ли данные для визуализации
        if len(self.numeric_cols) == 0 and len(self.categorical_cols) == 0:
            print("Нет данных для визуализации")
            return

        # Создаем фигуру с подграфиками 3x4
        fig = plt.figure(figsize=(20, 12))

        # 1. Гистограммы для числовых признаков (1-я строка)
        if self.numeric_cols:
            n_numeric = min(len(self.numeric_cols), 4)
            for i, col in enumerate(self.numeric_cols[:n_numeric]):
                ax = plt.subplot(3, 4, i + 1)
                self.data[col].hist(bins=30, edgecolor='black', alpha=0.7)
                plt.title(f'Распределение: {col}')
                plt.xlabel(str(col))
                plt.ylabel('Частота')
                plt.grid(True, alpha=0.3)

        # 2. Box plots для числовых признаков (2-я строка)
        if self.numeric_cols:
            n_numeric = min(len(self.numeric_cols), 4)
            for i, col in enumerate(self.numeric_cols[:n_numeric]):
                ax = plt.subplot(3, 4, i + 5)
                self.data.boxplot(column=col, ax=ax)
                plt.title(f'Box plot: {col}')
                plt.grid(True, alpha=0.3)

        # 3. Круговая диаграмма для категориальных данных (3-я строка, 1-й столбец)
        if self.categorical_cols:
            ax = plt.subplot(3, 4, 9)
            cat_col = self.categorical_cols[0]
            top_categories = self.data[cat_col].value_counts().head(5)
            if len(top_categories) > 0:
                top_categories.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                plt.title(f'Распределение: {cat_col}')
                plt.ylabel('')

        # 4. Тепловая карта корреляций (3-я строка, 2-й столбец)
        if len(self.numeric_cols) > 1:
            ax = plt.subplot(3, 4, 10)
            corr_matrix = self.data[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        ax=ax, cbar_kws={'label': 'Корреляция'})
            plt.title('Корреляционная матрица')

        plt.tight_layout()
        plt.show()

        # 5. Pairplot для числовых признаков (если их не太多)
        if 2 <= len(self.numeric_cols) <= 5:
            fig2 = plt.figure(figsize=(12, 10))
            pd.plotting.scatter_matrix(self.data[self.numeric_cols],
                                       alpha=0.5, figsize=(12, 12), diagonal='hist')
            plt.suptitle('Попарные зависимости числовых признаков', size=16)
            plt.tight_layout()
            plt.show()

    def statistical_analysis(self):
        """Статистический анализ данных"""
        if self.data is None:
            print("Ошибка: данные не загружены")
            return

        print("\n" + "=" * 80)
        print("СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ")
        print("=" * 80)

        # 1. Общая информация
        print("\n1. ОБЩАЯ ИНФОРМАЦИЯ:")
        print(f"  Количество записей: {len(self.data)}")
        print(f"  Количество признаков: {len(self.data.columns)}")
        print(f"  Количество пропусков: {self.data.isnull().sum().sum()}")
        print(f"  Количество дубликатов: {self.data.duplicated().sum()}")

        # 2. Статистика числовых признаков
        if self.numeric_cols:
            print("\n2. СТАТИСТИКА ЧИСЛОВЫХ ПРИЗНАКОВ:")
            stats_numeric = self.data[self.numeric_cols].describe()
            print(stats_numeric)

            # Дополнительные метрики
            print("\n   Дополнительные метрики:")
            for col in self.numeric_cols[:5]:
                print(f"\n   {col}:")
                print(f"     Дисперсия: {self.data[col].var():.2f}")
                print(f"     Асимметрия: {self.data[col].skew():.2f}")
                print(f"     Эксцесс: {self.data[col].kurtosis():.2f}")

        # 3. Статистика категориальных признаков
        if self.categorical_cols:
            print("\n3. СТАТИСТИКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:")
            for col in self.categorical_cols[:5]:
                print(f"\n   {col}:")
                print(f"     Уникальных значений: {self.data[col].nunique()}")
                print(f"     Наиболее частые значения:")
                print(self.data[col].value_counts().head(3))

        # 4. Анализ пропусков
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100

        missing_df = pd.DataFrame({
            'Пропуски': missing_data,
            'Процент': missing_percent
        }).sort_values('Процент', ascending=False)

        if missing_df['Пропуски'].sum() > 0:
            print("\n4. АНАЛИЗ ПРОПУСКОВ:")
            print(missing_df[missing_df['Пропуски'] > 0])

    def detect_specific_features(self):
        """Выявление специфических особенностей данных"""
        if self.data is None:
            print("Ошибка: данные не загружены")
            return

        print("\n" + "=" * 80)
        print("СПЕЦИФИЧЕСКИЕ ОСОБЕННОСТИ ДАННЫХ")
        print("=" * 80)

        # 1. Выбросы
        if self.numeric_cols:
            print("\n1. ВЫБРОСЫ (OUTLIERS):")
            outliers_found = False

            for col in self.numeric_cols:
                try:
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = self.data[(self.data[col] < lower_bound) |
                                         (self.data[col] > upper_bound)]

                    if len(outliers) > 0:
                        outliers_found = True
                        print(f"\n   {col}:")
                        print(
                            f"     Количество выбросов: {len(outliers)} ({len(outliers) / len(self.data) * 100:.1f}%)")
                        print(f"     Границы: [{lower_bound:.2f}, {upper_bound:.2f}]")
                        print(f"     Минимум: {self.data[col].min():.2f}, Максимум: {self.data[col].max():.2f}")
                except Exception as e:
                    print(f"\n   {col}: ошибка при вычислении - {e}")

            if not outliers_found:
                print("   Выбросов не обнаружено")

        # 2. Дисбаланс категорий
        if self.categorical_cols:
            print("\n2. ДИСБАЛАНС КАТЕГОРИЙ:")
            imbalance_found = False

            for col in self.categorical_cols:
                try:
                    value_counts = self.data[col].value_counts()

                    if len(value_counts) > 1:
                        max_count = value_counts.max()
                        min_count = value_counts.min()
                        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

                        if imbalance_ratio > 10:
                            imbalance_found = True
                            print(f"\n   {col}:")
                            print(f"     Коэффициент дисбаланса: {imbalance_ratio:.2f}")
                            print(f"     Самая частая категория: '{value_counts.index[0]}' ({max_count})")
                            print(f"     Самая редкая категория: '{value_counts.index[-1]}' ({min_count})")
                except Exception as e:
                    print(f"\n   {col}: ошибка при анализе - {e}")

            if not imbalance_found:
                print("   Значительного дисбаланса категорий не обнаружено")

        # 3. Мультиколлинеарность
        if len(self.numeric_cols) > 1:
            print("\n3. МУЛЬТИКОЛЛИНЕАРНОСТЬ:")
            try:
                corr_matrix = self.data[self.numeric_cols].corr()

                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': corr_val
                            })

                if high_corr_pairs:
                    print("\n   Пары с высокой корреляцией (>0.8):")
                    for pair in high_corr_pairs:
                        print(f"     {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
                else:
                    print("   Сильной мультиколлинеарности не обнаружено")
            except Exception as e:
                print(f"   Ошибка при вычислении корреляции: {e}")

        # 4. Константные столбцы
        print("\n4. КОНСТАНТНЫЕ СТОЛБЦЫ:")
        constant_cols = []
        for col in self.data.columns:
            try:
                if self.data[col].nunique() == 1:
                    constant_cols.append(str(col))
            except:
                pass

        if constant_cols:
            print(f"   Обнаружены константные столбцы: {', '.join(constant_cols)}")
            print("   Рекомендуется удалить их из анализа")
        else:
            print("   Константных столбцов не обнаружено")

        # 5. Качество данных
        print("\n5. КАЧЕСТВО ДАННЫХ:")

        # Проверка на нулевые значения в числовых столбцах
        zero_counts = {}
        for col in self.numeric_cols:
            try:
                zeros = (self.data[col] == 0).sum()
                if zeros > 0 and zeros < len(self.data):
                    zero_counts[col] = zeros
            except:
                pass

        if zero_counts:
            print("\n   Столбцы с нулевыми значениями:")
            for col, count in zero_counts.items():
                print(f"     {col}: {count} ({count / len(self.data) * 100:.1f}%)")

        # Проверка на пустые строки в текстовых столбцах
        empty_counts = {}
        for col in self.categorical_cols:
            try:
                empty = (self.data[col].astype(str).str.strip() == '').sum()
                if empty > 0:
                    empty_counts[col] = empty
            except:
                pass

        if empty_counts:
            print("\n   Столбцы с пустыми строками:")
            for col, count in empty_counts.items():
                print(f"     {col}: {count} ({count / len(self.data) * 100:.1f}%)")

    def generate_full_report(self):
        """Генерация полного отчета"""
        print("\n" + "=" * 80)
        print("ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ")
        print("=" * 80)

        self.statistical_analysis()
        self.detect_specific_features()
        self.visualize_data()

        print("\n" + "=" * 80)
        print("АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 80)


# ============================================
# 3. ОТОБРАЖЕНИЕ СПЕЦИФИКАЦИИ
# ============================================

def show_specification():
    """Отображение спецификации системы"""
    print("\n" + "=" * 80)
    print("СПЕЦИФИКАЦИЯ СИСТЕМЫ ПЕРВИЧНОГО АНАЛИЗА ДАННЫХ".center(80))
    print("=" * 80)
    print(f"\nВерсия: {__version__}")
    print(f"Описание: {__description__}")
    print("\nОсновные функции:")
    print("  1. Загрузка и визуализация данных")
    print("  2. Статистический анализ")
    print("  3. Выявление специфических особенностей")
    print("\nВыявляемые особенности:")
    print("  - Выбросы (outliers)")
    print("  - Дисбаланс категорий")
    print("  - Мультиколлинеарность")
    print("  - Константные столбцы")
    print("  - Пропуски и качество данных")
    print("\nПоддерживаемые форматы:")
    print("  - Стандартный CSV")
    print("  - CSV с данными в одном столбце")
    print("\nКоманды:")
    print("  exit - выход из программы")
    print("  spec - показать спецификацию")
    print("=" * 80)


# ============================================
# 4. ГЛАВНАЯ ФУНКЦИЯ
# ============================================

def main():
    """Основная функция"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--spec':
        show_specification()
        return

    print("\n" + "=" * 80)
    print("СИСТЕМА ПЕРВИЧНОГО АНАЛИЗА ТАБЛИЧНЫХ ДАННЫХ".center(80))
    print("=" * 80)
    print("\nДля просмотра спецификации запустите: python analyze.py --spec")

    analyzer = DataAnalyzer()

    csv_file = 'dataset.csv'
    if os.path.exists(csv_file):
        print(f"\nЗагрузка данных из {csv_file}")
        if not analyzer.load_data(csv_file):
            return
    else:
        print(f"\nФайл {csv_file} не найден")
        print("Пожалуйста, укажите правильный путь к файлу данных")
        return

    analyzer.generate_full_report()

    print("\nИнтерактивный режим (введите 'exit' для выхода, 'spec' для просмотра спецификации)")
    while True:
        user_input = input("\nВведите команду: ").strip()
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'spec':
            show_specification()
        elif user_input.lower() == 'stats':
            analyzer.statistical_analysis()
        elif user_input.lower() == 'features':
            analyzer.detect_specific_features()
        elif user_input.lower() == 'viz':
            analyzer.visualize_data()
        elif user_input:
            print("Неизвестная команда. Доступные команды: stats, features, viz, spec, exit")

    print("\nДо свидания!")


if __name__ == "__main__":
    main()