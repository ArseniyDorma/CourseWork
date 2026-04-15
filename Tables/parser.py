# -*- coding: utf-8 -*-
"""
Первичный анализ табличного набора данных для задачи Text-to-SQL
Датасет: text-to-sql (Kaggle)
Автор: Курсовой проект
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Настройка стилей графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ============================================================================
# 1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО С ДАННЫМИ
# ============================================================================

print("=" * 80)
print("1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО С ДАННЫМИ")
print("=" * 80)

# Загрузка данных (укажите актуальный путь к файлу)
df = pd.read_csv('dataset.csv')  # замените на путь к вашему файлу

print(f"\nФорма данных (строки, столбцы): {df.shape}")
print(f"\nПервые 5 строк данных:")
print(df.head())
print(f"\nТипы данных столбцов:")
print(df.dtypes)
print(f"\nНазвания столбцов: {df.columns.tolist()}")

# Создание производных числовых признаков для анализа
df['question_len'] = df['question'].astype(str).str.len()
df['query_len'] = df['query'].astype(str).str.len()
df['schema_len'] = df['schema'].astype(str).str.len()

# Подсчет количества ключевых слов в запросе
sql_keywords_list = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING',
                     'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT', 'UNION', 'CASE']


def count_keywords(query):
    query_upper = str(query).upper()
    return sum(1 for kw in sql_keywords_list if kw in query_upper)


df['keywords_count'] = df['query'].apply(count_keywords)

# ============================================================================
# 2. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================================

print("\n" + "=" * 80)
print("2. СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ")
print("=" * 80)

print("\nСтатистика текстовых признаков:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\nПризнак '{col}':")
        print(f"  - Количество уникальных значений: {df[col].nunique()}")
        print(f"  - Количество пустых строк: {df[col].isnull().sum()}")
        print(f"  - Примеры значений: {df[col].iloc[:2].tolist()}")

print("\nСтатистика производных числовых признаков:")
print(df[['question_len', 'query_len', 'schema_len', 'keywords_count']].describe())

# ============================================================================
# 3. АНАЛИЗ НА НАЛИЧИЕ ПРОПУСКОВ
# ============================================================================

print("\n" + "=" * 80)
print("3. АНАЛИЗ НА НАЛИЧИЕ ПРОПУСКОВ")
print("=" * 80)

missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_table = pd.DataFrame({
    'Количество пропусков': missing_data,
    'Доля пропусков, %': missing_percent
})
print(missing_table)

if missing_data.sum() == 0:
    print("\nПропущенные значения отсутствуют во всех столбцах.")

# Тепловая карта пропусков (если они есть)
if missing_data.sum() > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Рисунок 1. Карта пропусков в наборе данных', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png', dpi=150)
    plt.show()

# ============================================================================
# 4. УСТРАНЕНИЕ ДУБЛИКАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("4. УСТРАНЕНИЕ ДУБЛИКАТОВ")
print("=" * 80)

duplicates_count = df.duplicated().sum()
print(f"Количество полных дубликатов строк: {duplicates_count}")

if 'question' in df.columns and 'query' in df.columns:
    dup_question_query = df[['question', 'query']].duplicated().sum()
    print(f"Количество дубликатов по паре 'question-query': {dup_question_query}")

if duplicates_count > 0:
    df_cleaned = df.drop_duplicates()
    print(f"После удаления дубликатов осталось {len(df_cleaned)} строк")
else:
    df_cleaned = df.copy()

# ============================================================================
# 5. ДИАГРАММЫ РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ ПРИЗНАКОВ (Matplotlib)
# ============================================================================

print("\n" + "=" * 80)
print("5. ДИАГРАММЫ РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Длина вопроса
axes[0].hist(df['question_len'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(df['question_len'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Среднее: {df["question_len"].mean():.1f}')
axes[0].axvline(df['question_len'].median(), color='green', linestyle='dashed', linewidth=2,
                label=f'Медиана: {df["question_len"].median():.1f}')
axes[0].set_xlabel('Длина вопроса (символы)')
axes[0].set_ylabel('Частота')
axes[0].set_title('Распределение длин вопросов')
axes[0].legend()

# Длина запроса
axes[1].hist(df['query_len'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(df['query_len'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Среднее: {df["query_len"].mean():.1f}')
axes[1].axvline(df['query_len'].median(), color='green', linestyle='dashed', linewidth=2,
                label=f'Медиана: {df["query_len"].median():.1f}')
axes[1].set_xlabel('Длина SQL-запроса (символы)')
axes[1].set_ylabel('Частота')
axes[1].set_title('Распределение длин SQL-запросов')
axes[1].legend()

# Длина схемы
axes[2].hist(df['schema_len'], bins=50, edgecolor='black', alpha=0.7, color='seagreen')
axes[2].axvline(df['schema_len'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Среднее: {df["schema_len"].mean():.1f}')
axes[2].axvline(df['schema_len'].median(), color='green', linestyle='dashed', linewidth=2,
                label=f'Медиана: {df["schema_len"].median():.1f}')
axes[2].set_xlabel('Длина схемы (символы)')
axes[2].set_ylabel('Частота')
axes[2].set_title('Распределение длин описаний схем БД')
axes[2].legend()

# Количество ключевых слов
axes[3].hist(df['keywords_count'], bins=range(0, df['keywords_count'].max() + 2),
             edgecolor='black', alpha=0.7, color='purple')
axes[3].axvline(df['keywords_count'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Среднее: {df["keywords_count"].mean():.1f}')
axes[3].axvline(df['keywords_count'].median(), color='green', linestyle='dashed', linewidth=2,
                label=f'Медиана: {df["keywords_count"].median():.1f}')
axes[3].set_xlabel('Количество ключевых слов')
axes[3].set_ylabel('Частота')
axes[3].set_title('Распределение сложности SQL-запросов (по ключевым словам)')
axes[3].legend()

plt.suptitle('Рисунок 1. Диаграммы распределения числовых признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('numerical_distributions.png', dpi=150)
plt.show()

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ ПРИЗНАКОВ СРЕДСТВАМИ SEABORN (не менее 3 пар)
# ============================================================================

print("\n" + "=" * 80)
print("6. ВИЗУАЛИЗАЦИЯ ПРИЗНАКОВ СРЕДСТВАМИ SEABORN")
print("=" * 80)

# Пара 1: зависимость длины запроса от длины вопроса (scatterplot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(5000), x='question_len', y='query_len', alpha=0.5, hue='keywords_count',
                palette='viridis')
plt.xlabel('Длина вопроса (символы)')
plt.ylabel('Длина SQL-запроса (символы)')
plt.title('Рисунок 2. Зависимость длины SQL-запроса от длины вопроса')
plt.legend(title='Ключевых слов')
plt.tight_layout()
plt.savefig('seaborn_scatter.png', dpi=150)
plt.show()

# Пара 2: распределение длины запроса в зависимости от наличия JOIN (boxplot)
df['has_join'] = df['query'].astype(str).str.upper().str.contains('JOIN')
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='has_join', y='query_len', palette='Set2')
plt.xlabel('Наличие JOIN')
plt.ylabel('Длина SQL-запроса (символы)')
plt.title('Рисунок 3. Распределение длины запросов с JOIN и без JOIN')
plt.xticks([0, 1], ['Без JOIN', 'С JOIN'])
plt.tight_layout()
plt.savefig('seaborn_boxplot.png', dpi=150)
plt.show()

# Пара 3: плотность распределения длины вопроса для различных типов запросов (kdeplot)
plt.figure(figsize=(10, 6))
for has_join in [True, False]:
    subset = df[df['has_join'] == has_join]
    label = 'С JOIN' if has_join else 'Без JOIN'
    sns.kdeplot(data=subset, x='question_len', label=label, shade=True, alpha=0.5)
plt.xlabel('Длина вопроса (символы)')
plt.ylabel('Плотность')
plt.title('Рисунок 4. Плотность распределения длины вопросов в зависимости от наличия JOIN')
plt.legend()
plt.tight_layout()
plt.savefig('seaborn_kdeplot.png', dpi=150)
plt.show()

# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ ПРИЗНАКОВ СРЕДСТВАМИ PYPLOT (интерактивные графики)
# ============================================================================

print("\n" + "=" * 80)
print("7. ВИЗУАЛИЗАЦИЯ ПРИЗНАКОВ СРЕДСТВАМИ PYPLOT")
print("=" * 80)

# Интерактивный график 1: накопленная частота длин вопросов
plt.figure(figsize=(10, 6))
sorted_lengths = np.sort(df['question_len'])
cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
plt.plot(sorted_lengths, cumulative, linewidth=2, color='steelblue')
plt.axhline(y=80, color='red', linestyle='dashed', linewidth=1.5, label='80% вопросов')
plt.axhline(y=95, color='green', linestyle='dashed', linewidth=1.5, label='95% вопросов')
plt.xlabel('Длина вопроса (символы)')
plt.ylabel('Накопленная частота, %')
plt.title('Рисунок 5. Интерактивный график накопленной частоты длин вопросов')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('pyplot_cumulative.png', dpi=150)
plt.show()

# Интерактивный график 2: зависимость сложности запроса от длины вопроса
plt.figure(figsize=(10, 6))
plt.scatter(df['question_len'], df['keywords_count'], alpha=0.3, c='coral', edgecolors='none')
z = np.polyfit(df['question_len'], df['keywords_count'], 1)
p = np.poly1d(z)
plt.plot(df['question_len'].sort_values(), p(df['question_len'].sort_values()),
         'r-', linewidth=2, label=f'Тренд (y={z[0]:.2f}x+{z[1]:.2f})')
plt.xlabel('Длина вопроса (символы)')
plt.ylabel('Количество ключевых слов в SQL-запросе')
plt.title('Рисунок 6. Зависимость сложности запроса от длины вопроса')
plt.legend()
plt.tight_layout()
plt.savefig('pyplot_trend.png', dpi=150)
plt.show()

# ============================================================================
# 8. ПОСТРОЕНИЕ ТЕПЛОВЫХ КАРТ (не менее 2)
# ============================================================================

print("\n" + "=" * 80)
print("8. ПОСТРОЕНИЕ ТЕПЛОВЫХ КАРТ")
print("=" * 80)

# Тепловая карта 1: корреляции между числовыми признаками
numeric_features = ['question_len', 'query_len', 'schema_len', 'keywords_count']
corr_matrix = df[numeric_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={'shrink': 0.8})
plt.title('Рисунок 7. Тепловая карта корреляций между числовыми признаками', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_correlation.png', dpi=150)
plt.show()

# Тепловая карта 2: совместная встречаемость ключевых слов SQL
top_keywords = ['WHERE', 'GROUP BY', 'ORDER BY', 'JOIN', 'COUNT', 'AVG', 'SUM']
keyword_matrix = pd.DataFrame(index=top_keywords, columns=top_keywords, dtype=float)
for kw1 in top_keywords:
    for kw2 in top_keywords:
        mask1 = df['query'].astype(str).str.upper().str.contains(kw1)
        mask2 = df['query'].astype(str).str.upper().str.contains(kw2)
        co_occurrence = (mask1 & mask2).sum()
        total = mask1.sum()
        keyword_matrix.loc[kw1, kw2] = co_occurrence / total if total > 0 else 0

plt.figure(figsize=(10, 8))
sns.heatmap(keyword_matrix, annot=True, cmap='YlOrRd', fmt='.2f',
            square=True, linewidths=0.5, cbar_kws={'label': 'Частота совместной встречаемости'})
plt.title('Рисунок 8. Тепловая карта совместной встречаемости ключевых слов SQL', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_keywords.png', dpi=150)
plt.show()

# ============================================================================
# 9. АНАЛИЗ ВЫБРОСОВ ПО ПРИЗНАКАМ
# ============================================================================

print("\n" + "=" * 80)
print("9. АНАЛИЗ ВЫБРОСОВ ПО ПРИЗНАКАМ")
print("=" * 80)

outliers_summary = {}
for col in ['question_len', 'query_len', 'schema_len', 'keywords_count']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_summary[col] = {
        'нижняя_граница': lower_bound,
        'верхняя_граница': upper_bound,
        'количество_выбросов': len(outliers),
        'доля_выбросов': len(outliers) / len(df) * 100
    }
    print(f"\n{col}:")
    print(f"  - Верхняя граница IQR: {upper_bound:.1f}")
    print(f"  - Выбросов: {len(outliers)} ({len(outliers) / len(df) * 100:.2f}%)")

# ============================================================================
# 10. УСЛОВНАЯ ФИЛЬТРАЦИЯ СЭМПЛОВ (не менее 3 различных фильтраций)
# ============================================================================

print("\n" + "=" * 80)
print("10. УСЛОВНАЯ ФИЛЬТРАЦИЯ СЭМПЛОВ")
print("=" * 80)

# Фильтрация 1: запросы с JOIN
filter1 = df[df['query'].astype(str).str.upper().str.contains('JOIN')]
print(f"\nФильтрация 1 (запросы с JOIN): {len(filter1)} записей ({len(filter1) / len(df) * 100:.1f}%)")

# Фильтрация 2: запросы с агрегатными функциями
agg_pattern = 'COUNT|SUM|AVG|MAX|MIN'
filter2 = df[df['query'].astype(str).str.upper().str.contains(agg_pattern)]
print(f"Фильтрация 2 (запросы с агрегатными функциями): {len(filter2)} записей ({len(filter2) / len(df) * 100:.1f}%)")

# Фильтрация 3: длинные вопросы (> 100 символов) И сложные запросы (> 5 ключевых слов)
filter3 = df[(df['question_len'] > 100) & (df['keywords_count'] > 5)]
print(f"Фильтрация 3 (длинные вопросы И сложные запросы): {len(filter3)} записей ({len(filter3) / len(df) * 100:.2f}%)")

# Демонстрация результатов фильтрации
print("\nПримеры из фильтрации 1 (первые 3 запроса с JOIN):")
for i, query in enumerate(filter1['query'].head(3)):
    print(f"  {i + 1}. {query[:100]}...")

# ============================================================================
# 11. ДОБАВЛЕНИЕ ШУМА (не менее чем в 2 признака, исключая целевые)
# ============================================================================

print("\n" + "=" * 80)
print("11. ДОБАВЛЕНИЕ ШУМА")
print("=" * 80)

df_noisy = df.copy()


# Шум в признак 'question' (символьный шум: замена 3% символов)
def add_character_noise(text, noise_level=0.03):
    text = str(text)
    if len(text) == 0:
        return text
    chars = list(text)
    num_noisy = max(1, int(len(chars) * noise_level))
    indices = np.random.choice(len(chars), num_noisy, replace=False)
    for idx in indices:
        chars[idx] = np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    return ''.join(chars)


np.random.seed(42)
sample_indices = np.random.choice(len(df_noisy), min(1000, len(df_noisy)), replace=False)
df_noisy.loc[sample_indices, 'question_noisy'] = df_noisy.loc[sample_indices, 'question'].apply(add_character_noise)
df_noisy['question_noisy'] = df_noisy['question_noisy'].fillna(df_noisy['question'])


# Шум в признак 'query' (изменение регистра ключевых слов)
def add_case_noise(query):
    query = str(query)
    keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY']
    for kw in keywords:
        if kw in query.upper():
            noisy_kw = ''.join([c.lower() if np.random.random() > 0.5 else c.upper() for c in kw])
            query = query.replace(kw, noisy_kw, 1)
    return query


df_noisy.loc[sample_indices, 'query_noisy'] = df_noisy.loc[sample_indices, 'query'].apply(add_case_noise)
df_noisy['query_noisy'] = df_noisy['query_noisy'].fillna(df_noisy['query'])

# Визуализация влияния шума
plt.figure(figsize=(12, 6))
plt.hist(df['question_len'], bins=50, alpha=0.5, label='Исходные вопросы', color='steelblue')
plt.hist(df_noisy['question_noisy'].str.len(), bins=50, alpha=0.5, label='Зашумленные вопросы', color='coral')
plt.xlabel('Длина вопроса (символы)')
plt.ylabel('Частота')
plt.title('Рисунок 9. Влияние добавления шума на распределение длин вопросов')
plt.legend()
plt.tight_layout()
plt.savefig('noise_effect.png', dpi=150)
plt.show()

print("Шум добавлен в 1000 записей (признаки 'question' и 'query')")
print("Пример исходного вопроса:", df['question'].iloc[0][:100])
print("Пример зашумленного вопроса:", df_noisy['question_noisy'].iloc[0][:100])

# ============================================================================
# 12. ПРЕОБРАЗОВАНИЕ ЧИСЛОВЫХ ДАННЫХ В КАТЕГОРИАЛЬНЫЕ
# ============================================================================

print("\n" + "=" * 80)
print("12. ПРЕОБРАЗОВАНИЕ ЧИСЛОВЫХ ДАННЫХ В КАТЕГОРИАЛЬНЫЕ")
print("=" * 80)


# Категоризация вопросов по длине
def categorize_question_length(length):
    if length < 50:
        return 'короткий (<50)'
    elif length < 100:
        return 'средний (50-100)'
    else:
        return 'длинный (>100)'


df['question_category'] = df['question_len'].apply(categorize_question_length)
print("\nРаспределение категорий вопросов:")
print(df['question_category'].value_counts())


# Категоризация запросов по сложности
def categorize_query_complexity(keywords_count):
    if keywords_count <= 3:
        return 'простой'
    elif keywords_count <= 6:
        return 'средний'
    else:
        return 'сложный'


df['complexity_category'] = df['keywords_count'].apply(categorize_query_complexity)
print("\nРаспределение категорий сложности запросов:")
print(df['complexity_category'].value_counts())

# ============================================================================
# 13. ДОПОЛНИТЕЛЬНЫЕ ПРЕОБРАЗОВАНИЯ (унификация обозначений)
# ============================================================================

print("\n" + "=" * 80)
print("13. ДОПОЛНИТЕЛЬНЫЕ ПРЕОБРАЗОВАНИЯ")
print("=" * 80)

# Унификация регистра SQL-запросов
df['query_normalized'] = df['query'].astype(str).str.upper()
print("SQL-запросы приведены к верхнему регистру")

# Унификация типов данных в схеме
df['schema_normalized'] = df['schema'].astype(str).str.replace('INTEGER', 'INT')
df['schema_normalized'] = df['schema_normalized'].str.replace('TEXT', 'VARCHAR')
print("Типы данных в схеме унифицированы (INTEGER→INT, TEXT→VARCHAR)")

# ============================================================================
# 14. ОЦЕНКА ИЗМЕНЕНИЯ В ДАННЫХ ПОСЛЕ ФИЛЬТРАЦИИ (повторная визуализация)
# ============================================================================

print("\n" + "=" * 80)
print("14. ОЦЕНКА ИЗМЕНЕНИЯ В ДАННЫХ ПОСЛЕ ФИЛЬТРАЦИИ")
print("=" * 80)

# Сравнение распределений до и после фильтрации (удаление выбросов)
df_no_outliers = df[(df['question_len'] <= 200) & (df['query_len'] <= 400)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['question_len'], bins=50, alpha=0.5, label='Исходные данные', color='steelblue')
axes[0].hist(df_no_outliers['question_len'], bins=50, alpha=0.5, label='После удаления выбросов', color='coral')
axes[0].set_xlabel('Длина вопроса (символы)')
axes[0].set_ylabel('Частота')
axes[0].set_title('Распределение длин вопросов')
axes[0].legend()

axes[1].hist(df['query_len'], bins=50, alpha=0.5, label='Исходные данные', color='steelblue')
axes[1].hist(df_no_outliers['query_len'], bins=50, alpha=0.5, label='После удаления выбросов', color='coral')
axes[1].set_xlabel('Длина SQL-запроса (символы)')
axes[1].set_ylabel('Частота')
axes[1].set_title('Распределение длин запросов')
axes[1].legend()

plt.suptitle('Рисунок 10. Сравнение распределений до и после фильтрации выбросов', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('filtering_effect.png', dpi=150)
plt.show()

print(f"После удаления выбросов (question_len<=200, query_len<=400):")
print(f"  - Исходное количество записей: {len(df)}")
print(f"  - После фильтрации: {len(df_no_outliers)}")
print(f"  - Удалено записей: {len(df) - len(df_no_outliers)} ({(len(df) - len(df_no_outliers)) / len(df) * 100:.2f}%)")

# ============================================================================
# 15. ЗАПРОСЫ НА ГРУППИРОВКУ (для составных данных - топ-10 схем)
# ============================================================================

print("\n" + "=" * 80)
print("15. ЗАПРОСЫ НА ГРУППИРОВКУ")
print("=" * 80)

# Группировка по схемам баз данных
schema_counts = df.groupby('schema').size().reset_index(name='count')
schema_counts = schema_counts.sort_values('count', ascending=False).head(10)

print("\nТоп-10 схем баз данных по частоте использования:")
import re

# Вывод с понятными названиями
for idx, row in schema_counts.iterrows():
    # Извлекаем название таблицы из CREATE-запроса
    match = re.search(r'CREATE TABLE (\w+)', row['schema'], re.IGNORECASE)
    if match:
        table_name = match.group(1)
        print(f"  - {table_name}: {row['count']} запросов")
    else:
        preview = row['schema'].replace('\n', ' ').replace('\t', ' ')[:40] + "..."
        print(f"  - {preview}: {row['count']} запросов")

# Визуализация группировки с осмысленными подписями
plt.figure(figsize=(14, 8))

short_names = []
name_counts = {}
for schema in schema_counts['schema'].values:
    match = re.search(r'CREATE TABLE (\w+)', schema, re.IGNORECASE)
    if match:
        table_name = match.group(1)
        # Обработка повторяющихся названий
        if table_name in name_counts:
            name_counts[table_name] += 1
            short_names.append(f"{table_name} ({name_counts[table_name]})")
        else:
            name_counts[table_name] = 1
            short_names.append(table_name)
    else:
        short_names.append("unknown")

plt.barh(range(len(schema_counts)), schema_counts['count'], color='steelblue')
plt.yticks(range(len(schema_counts)), short_names)
plt.xlabel('Количество запросов', fontsize=12)
plt.ylabel('Название таблицы (основная таблица в схеме)', fontsize=12)
plt.title('Рисунок 11. Топ-10 схем баз данных по частоте использования', fontsize=14, fontweight='bold')

for i, v in enumerate(schema_counts['count']):
    plt.text(v + 10, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('grouping_top_schemas.png', dpi=150)
plt.show()

# ============================================================================
# 16. ХАРАКТЕРИСТИКА КАТЕГОРИАЛЬНЫХ ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("16. ХАРАКТЕРИСТИКА КАТЕГОРИАЛЬНЫХ ДАННЫХ")
print("=" * 80)

# Перечень категорий для has_join
print("\nКатегории признака 'has_join':")
print(f"  - Без JOIN: {len(df[df['has_join'] == False])} записей")
print(f"  - С JOIN: {len(df[df['has_join'] == True])} записей")

# Диаграмма распределения категориальных данных
plt.figure(figsize=(8, 6))
category_counts = df['complexity_category'].value_counts()
plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
        startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Рисунок 12. Распределение запросов по категориям сложности')
plt.tight_layout()
plt.savefig('categorical_pie.png', dpi=150)
plt.show()

# Преобразование категориальных данных в числовые (Label Encoding для has_join)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['has_join_encoded'] = le.fit_transform(df['has_join'])
print("\nПреобразование 'has_join' в числовой формат (Label Encoding):")
print(f"  - False → {le.transform([False])[0]}")
print(f"  - True → {le.transform([True])[0]}")

# ============================================================================
# 17. АГРЕГАЦИЯ ДАННЫХ (редкие категории)
# ============================================================================

print("\n" + "=" * 80)
print("17. АГРЕГАЦИЯ ДАННЫХ (РЕДКИЕ КАТЕГОРИИ)")
print("=" * 80)

# Анализ редких схем (встречаемость менее 10 раз)
schema_counts_all = df.groupby('schema').size()
rare_schemas = schema_counts_all[schema_counts_all < 10]
print(f"Количество редких схем (встречаемость < 10): {len(rare_schemas)}")
print(f"Доля редких схем от всех уникальных схем: {len(rare_schemas) / len(schema_counts_all) * 100:.1f}%")

# Агрегация: объединение редких схем в категорию 'other'
df['schema_aggregated'] = df['schema'].apply(
    lambda x: x if schema_counts_all[x] >= 10 else 'other_schema'
)
print(f"\nПосле агрегации уникальных схем: {df['schema_aggregated'].nunique()}")

# ============================================================================
# 18. ВВЕДЕНИЕ НОВОЙ КАТЕГОРИИ (на основе нескольких признаков)
# ============================================================================

print("\n" + "=" * 80)
print("18. ВВЕДЕНИЕ НОВОЙ КАТЕГОРИИ")
print("=" * 80)


# Новая категория: уровень сложности запроса (на основе JOIN, GROUP BY, агрегаций)
def determine_query_level(query):
    query_upper = str(query).upper()
    has_join = 'JOIN' in query_upper
    has_groupby = 'GROUP BY' in query_upper
    has_agg = any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN'])

    if has_join and has_groupby and has_agg:
        return 'very_high'
    elif has_join or (has_groupby and has_agg):
        return 'high'
    elif has_groupby or has_agg:
        return 'medium'
    else:
        return 'low'


df['query_level'] = df['query'].apply(determine_query_level)
print("\nРаспределение новой категории 'query_level':")
print(df['query_level'].value_counts())

# Визуализация новой категории
plt.figure(figsize=(10, 6))
level_order = ['low', 'medium', 'high', 'very_high']
level_counts = df['query_level'].value_counts().reindex(level_order)
plt.bar(level_counts.index, level_counts.values, color=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad'])
plt.xlabel('Уровень сложности запроса')
plt.ylabel('Количество запросов')
plt.title('Рисунок 13. Распределение запросов по уровню сложности (новая категория)')
for i, v in enumerate(level_counts.values):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('new_category.png', dpi=150)
plt.show()

# ============================================================================
# 19. СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("19. СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ")
print("=" * 80)

# Сохранение обогащенного датасета
df.to_csv('dataset_enhanced.csv', index=False)
print("Обогащенный датасет сохранен как 'train_enhanced.csv'")

print("\n" + "=" * 80)
print("Анализ завершен. Сохранены следующие файлы:")
print("  - numerical_distributions.png (диаграммы распределения)")
print("  - seaborn_scatter.png, seaborn_boxplot.png, seaborn_kdeplot.png")
print("  - pyplot_cumulative.png, pyplot_trend.png")
print("  - heatmap_correlation.png, heatmap_keywords.png")
print("  - noise_effect.png, filtering_effect.png")
print("  - grouping_top_schemas.png, categorical_pie.png, new_category.png")
print("  - train_enhanced.csv (обогащенный датасет)")
print("=" * 80)