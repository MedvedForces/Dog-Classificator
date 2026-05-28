import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 224  
BATCH_SIZE = 32  
EPOCHS = 20


labels_df = pd.read_csv('labels.csv')
print(f"Всего изображений: {len(labels_df)}")
print(f"Всего пород: {labels_df['breed'].nunique()}")

# Кодирование пород
le = LabelEncoder()
labels_df['label'] = le.fit_transform(labels_df['breed'])
num_classes = len(le.classes_)
print(f"Количество классов: {num_classes}")

# Разделение на train/val
train_df, val_df = train_test_split(
    labels_df, test_size=0.2, random_state=42, 
    stratify=labels_df['label']  # важно для баланса классов
)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")


# Функция загрузки изображений

def load_and_preprocess(image_id, label):
    path = tf.strings.join(['train/', image_id, '.jpg'])
    
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

# Случайным образом меняем изображение для увеличения датасета 
def augment(image, label):
    # горизонтальное отражение
    image = tf.image.random_flip_left_right(image)
    # Изменение яркости
    image = tf.image.random_brightness(image, 0.1)
    # Контраст
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    return image, label

# Создаем датасеты
def create_dataset(df, batch_size, training=True):
    ids = df['id'].values
    labels = tf.keras.utils.to_categorical(df['label'].values, num_classes)
    
    dataset = tf.data.Dataset.from_tensor_slices((ids, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

train_ds = create_dataset(train_df, BATCH_SIZE, training=True)
val_ds = create_dataset(val_df, BATCH_SIZE, training=False)


# Создание модели CNN

print("\nСоздание модели...")
model = tf.keras.Sequential([
    # Сверточные слои: (нахождение признаков)
    # от 32 до 512 слоев
    # нули по краям, чтобы не резать (padding='same')
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    # Нормализуем данные, чтобы не было переполнения 
    tf.keras.layers.BatchNormalization(),  
    # Достаем из матрицы блоки и берем из них максимумы (уменьшаем матрицу)
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    
    # Принимаем 2D, переводим в вектор с помощью среднего 
    tf.keras.layers.GlobalAveragePooling2D(), 
    
    # Полносвязные слои: (принятие решения на основе признаков)
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Компиляция с хорошими параметрами
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# Callbacks

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_dog_breed_model.h5', 
        monitor='val_accuracy', 
        save_best_only=True,
        verbose=1
    )
]


# Обучение

print("\nНачинаем обучение...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# Предсказание для теста

print("\nПредсказание для тестовых изображений...")
test_ids = [f.replace('.jpg', '') for f in os.listdir('test') if f.endswith('.jpg')]
print(f"Тестовых изображений: {len(test_ids)}")

def predict_test(model, test_ids, batch_size=32):
    predictions = []
    for i in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[i:i+batch_size]
        batch_images = []
        
        for img_id in batch_ids:
            path = f'test/{img_id}.jpg'
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
            image = tf.cast(image, tf.float32) / 255.0
            batch_images.append(image)
        
        batch_preds = model.predict(np.array(batch_images), verbose=0)
        predictions.extend(batch_preds)
        
        if (i // batch_size) % 10 == 0:
            print(f"Обработано {i+len(batch_ids)} из {len(test_ids)}...")
    
    return np.array(predictions)

test_predictions = predict_test(model, test_ids)


# Сохранение результата

print("\nСохранение submission файла...")
sample_sub = pd.read_csv('sample_submission.csv')

submission = pd.DataFrame({'id': test_ids})

# Добавляем вероятности для каждой породы
for i, breed in enumerate(le.classes_):
    submission[breed] = test_predictions[:, i]

# Убеждаемся, что колонки в правильном порядке
submission = submission[sample_sub.columns]

# Сохраняем
submission.to_csv('submission.csv', index=False)
print("Готово! Файл submission.csv создан.")
print(f"Первые 5 строк:")
print(submission.head())
