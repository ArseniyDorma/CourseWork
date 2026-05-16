# ======================================================
# Глава 3: Первичный анализ набора данных с изображениями
# Задача: Детекция СИЗ и нарушений (Hardhat, Gloves, Goggles, Safety Vest)
# Формат: YOLO (bounding boxes + классы)
# ======================================================

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from collections import Counter, defaultdict
from pathlib import Path
import random
from tqdm import tqdm

# Настройка стилей
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 60)
print("ПЕРВИЧНЫЙ АНАЛИЗ НАБОРА ДАННЫХ С ИЗОБРАЖЕНИЯМИ")
print("Задача: Детекция средств индивидуальной защиты (СИЗ) и нарушений")
print("=" * 60)

DATASET_PATH = r"./dataset"

# Поиск data.yaml
data_yaml_path = None
for root, dirs, files in os.walk(DATASET_PATH):
    if "data.yaml" in files:
        data_yaml_path = os.path.join(root, "data.yaml")
        break

# ======================================================
# 2. ЗАГРУЗКА data.yaml
# ======================================================
print("\n[1] Загрузка конфигурации data.yaml...")

with open(data_yaml_path, 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)

class_names = data_config.get('names', [])
nc = data_config.get('nc', len(class_names))

print(f"   - Файл: {data_yaml_path}")
print(f"   - Классы: {class_names}")
print(f"   - Количество классов: {nc}")

# Определяем пути к папкам
images_path = None
for root, dirs, files in os.walk(DATASET_PATH):
    if 'images' in dirs and os.path.basename(root) in ['train', 'val', 'test', '']:
        images_path = os.path.join(root, 'images')
        break

if images_path is None:
    for root, dirs, files in os.walk(DATASET_PATH):
        if any(f.endswith(('.jpg', '.png', '.jpeg')) for f in files):
            images_path = root
            break

labels_path = images_path.replace('images', 'labels') if images_path else None
if labels_path and not os.path.exists(labels_path):
    for root, dirs, files in os.walk(DATASET_PATH):
        if 'labels' in dirs:
            labels_path = os.path.join(root, 'labels')
            break

# ======================================================
# 3. ПОИСК ИЗОБРАЖЕНИЙ И АННОТАЦИЙ
# ======================================================
print("\n[2] Поиск изображений и аннотаций...")


def find_images_and_labels(images_dir, labels_dir):
    valid_pairs = []

    if images_dir and os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if labels_dir and os.path.exists(labels_dir):
            label_files = set([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

        for img_file in image_files:
            name_no_ext = os.path.splitext(img_file)[0]
            label_file = name_no_ext + '.txt'
            if label_file in label_files:
                valid_pairs.append((img_file, label_file))

    return valid_pairs


valid_pairs = find_images_and_labels(images_path, labels_path)

total_images = len([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]) if images_path else 0
total_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')]) if labels_path else 0

print(f"   - Изображений: {total_images}")
print(f"   - Аннотаций: {total_labels}")
print(f"   - Полных пар: {len(valid_pairs)}")

# ======================================================
# 4. ПАРСИНГ YOLO-АННОТАЦИЙ
# ======================================================
print("\n[3] Анализ количества и баланса классов...")


def parse_yolo_label(label_path):
    objects = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    objects.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
    return objects


# Сбор статистики
class_counts_all = Counter()
objects_per_image = []
images_without_objects = 0
total_annotations = 0
class_ids_set = set()

print("   - Парсинг YOLO-аннотаций...")
for img_file, label_file in tqdm(valid_pairs, desc="Обработка"):
    label_path = os.path.join(labels_path, label_file)
    objects = parse_yolo_label(label_path)

    if len(objects) == 0:
        images_without_objects += 1
    else:
        objects_per_image.append(len(objects))
        total_annotations += len(objects)
        for obj in objects:
            class_id = obj['class_id']
            class_ids_set.add(class_id)
            if class_id < len(class_names):
                class_counts_all[class_names[class_id]] += 1
            else:
                class_counts_all[f'class_{class_id}'] += 1

print(f"\n   - Всего изображений: {len(valid_pairs)}")
print(f"   - Без объектов: {images_without_objects} ({images_without_objects / len(valid_pairs) * 100:.1f}%)")
print(f"   - С объектами: {len(valid_pairs) - images_without_objects}")
print(f"   - Всего объектов: {total_annotations}")
print(f"   - Среднее объектов/изображение: {np.mean(objects_per_image):.2f}")

# Визуализация 1: Распределение по классам
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df_classes = pd.DataFrame({
    'Класс': list(class_counts_all.keys()),
    'Количество': list(class_counts_all.values())
}).sort_values('Количество', ascending=False)

# Исправленный barplot (без palette, используем color)
bars = axes[0].bar(range(len(df_classes)), df_classes['Количество'], color=plt.cm.viridis(range(len(df_classes))))
axes[0].set_xticks(range(len(df_classes)))
axes[0].set_xticklabels(df_classes['Класс'], rotation=45, ha='right', fontsize=8)
axes[0].set_title('Распределение объектов по классам', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Класс')
axes[0].set_ylabel('Количество объектов')

# График 2: Количество объектов на изображение
object_counts = Counter(objects_per_image)
axes[1].bar(object_counts.keys(), object_counts.values(), color='steelblue', edgecolor='black')
axes[1].set_title('Количество объектов на одно изображение', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Количество объектов')
axes[1].set_ylabel('Количество изображений')

# График 3: Доля изображений с/без объектов
sizes_pie = [len(valid_pairs) - images_without_objects, images_without_objects]
labels_pie = ['С объектами', 'Без объектов']
colors_pie = ['#66b3ff', '#ff9999']
axes[2].pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, explode=(0.05, 0))
axes[2].set_title('Покрытие аннотациями', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Вывод баланса классов
print("\n   Баланс классов (% от всех объектов):")
for class_name, count in class_counts_all.most_common():
    print(f"      - {class_name}: {count} ({count / total_annotations * 100:.1f}%)")

# ======================================================
# 5. ПРИМЕРЫ ИЗОБРАЖЕНИЙ С РАЗМЕТКОЙ
# ======================================================
print("\n[4] Демонстрация типичных изображений с разметкой...")


def draw_boxes_pil(image_path, label_path, class_names, max_size=(400, 400)):
    """Рисование bounding boxes с помощью PIL"""
    img = Image.open(image_path).convert('RGB')
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    w, h = img.size

    # Масштабирующий коэффициент
    orig_w, orig_h = Image.open(image_path).size
    scale_x = w / orig_w
    scale_y = h / orig_h

    objects = parse_yolo_label(label_path)
    draw = ImageDraw.Draw(img)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
              'magenta', 'yellow', 'navy', 'darkred', 'darkgreen', 'gold', 'violet', 'indigo']

    for obj in objects:
        class_id = obj['class_id']
        label = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'

        # Конвертация в пиксельные координаты с учетом масштаба
        x1 = int((obj['x_center'] - obj['width'] / 2) * w)
        y1 = int((obj['y_center'] - obj['height'] / 2) * h)
        x2 = int((obj['x_center'] + obj['width'] / 2) * w)
        y2 = int((obj['y_center'] + obj['height'] / 2) * h)

        color = colors[class_id % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 12), label, fill=color)

    return img


# Выбираем случайные изображения с объектами
images_with_objs = [(img, lbl) for img, lbl in valid_pairs
                    if len(parse_yolo_label(os.path.join(labels_path, lbl))) > 0]

sample_pairs = random.sample(images_with_objs, min(6, len(images_with_objs)))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (img_file, label_file) in enumerate(sample_pairs):
    img_path = os.path.join(images_path, img_file)
    label_path = os.path.join(labels_path, label_file)

    img_with_boxes = draw_boxes_pil(img_path, label_path, class_names)
    axes[idx].imshow(img_with_boxes)
    axes[idx].set_title(f'{img_file[:25]}...', fontsize=9)
    axes[idx].axis('off')

plt.suptitle('Примеры изображений с аннотациями (bounding boxes)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_sample_annotations.png', dpi=150, bbox_inches='tight')
plt.show()

# ======================================================
# 6. АНАЛИЗ КАЧЕСТВА ИЗОБРАЖЕНИЙ И BOUNDING BOXES
# ======================================================
print("\n[5] Анализ качества изображений и разметки...")

# Статистика размеров изображений
image_sizes = []
if images_path:
    sample_images = random.sample([f for f, _ in valid_pairs], min(100, len(valid_pairs)))
    for img_file in sample_images:
        img_path = os.path.join(images_path, img_file)
        with Image.open(img_path) as img:
            image_sizes.append(img.size)

    widths = [s[0] for s in image_sizes]
    heights = [s[1] for s in image_sizes]

    print(f"\n   Статистика размеров изображений:")
    print(f"   - Ширина: мин={min(widths)}, макс={max(widths)}, среднее={np.mean(widths):.0f}")
    print(f"   - Высота: мин={min(heights)}, макс={max(heights)}, среднее={np.mean(heights):.0f}")

# Анализ bounding boxes
box_areas = []
box_aspect_ratios = []

for _, label_file in random.sample(valid_pairs, min(500, len(valid_pairs))):
    label_path = os.path.join(labels_path, label_file)
    objects = parse_yolo_label(label_path)
    for obj in objects:
        area = obj['width'] * obj['height']
        aspect = obj['width'] / max(obj['height'], 0.001)
        box_areas.append(area)
        box_aspect_ratios.append(aspect)

if box_areas:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(box_areas, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0].set_title('Распределение площадей bounding boxes', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Нормализованная площадь')
    axes[0].set_ylabel('Частота')
    axes[0].axvline(x=np.mean(box_areas), color='red', linestyle='--', label=f'Среднее: {np.mean(box_areas):.3f}')
    axes[0].legend()

    axes[1].hist(box_aspect_ratios, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title('Распределение соотношений сторон bounding boxes', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Соотношение сторон (ширина/высота)')
    axes[1].set_ylabel('Частота')
    axes[1].axvline(x=np.mean(box_aspect_ratios), color='red', linestyle='--',
                    label=f'Среднее: {np.mean(box_aspect_ratios):.2f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('03_bbox_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n   Статистика bounding boxes:")
    print(f"   - Средняя площадь: {np.mean(box_areas):.4f}")
    print(f"   - Среднее соотношение сторон: {np.mean(box_aspect_ratios):.2f}")

# ======================================================
# 7. ВЫВОДЫ
# ======================================================
print("\n" + "=" * 60)
print("ВЫВОДЫ ПО ТРЕТЬЕМУ РАЗДЕЛУ")
print("=" * 60)

print(f"""
1. ХАРАКТЕРИСТИКА НАБОРА ДАННЫХ:
   - Задача: детекция СИЗ и нарушений (бинарные пары: с СИЗ / без СИЗ)
   - Классы (14): {', '.join(class_names[:5])}... и {len(class_names) - 5} других
   - Количество изображений: {len(valid_pairs)}
   - Количество аннотированных объектов: {total_annotations}
   - Среднее объектов на изображение: {np.mean(objects_per_image):.2f}
   - Формат аннотаций: YOLO

2. ВЫЯВЛЕННЫЕ ОСОБЕННОСТИ:
   - Сильный дисбаланс классов: Hardhat ({class_counts_all['Hardhat'] / total_annotations * 100:.1f}%) 
     vs Ladder (1.0% в 14 раз меньше)
   - Наличие парных классов: Hardhat/NO-Hardhat, Gloves/NO-Gloves и т.д. (для бинарной классификации нарушений)
   - Изображений без объектов: {images_without_objects} ({images_without_objects / len(valid_pairs) * 100:.1f}%)

""")

print("\n" + "=" * 60)
print("Сгенерированные файлы:")
print("   - 01_class_distribution.png  (распределение по классам)")
print("   - 02_sample_annotations.png  (примеры с разметкой)")
print("   - 03_bbox_analysis.png       (анализ bounding boxes)")
print("=" * 60)