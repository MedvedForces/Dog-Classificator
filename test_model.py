# test_model.py
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# ============ ПАРАМЕТРЫ ============
IMG_SIZE = 224
BATCH_SIZE = 32

# ============ 1. ЗАГРУЗКА МОДЕЛИ ============
print("Загрузка модели...")
model = tf.keras.models.load_model('best_dog_breed_model.h5')
print("Модель загружена успешно!")

# ============ 2. ЗАГРУЗКА НАЗВАНИЙ ПОРОД ИЗ CSV ============
print("Загрузка названий пород...")
sample_sub = pd.read_csv('sample_submission.csv')
breed_names = sample_sub.columns[1:].tolist()  # все колонки кроме 'id'
print(f"Загружено {len(breed_names)} пород")

# ============ 3. ФУНКЦИЯ ЗАГРУЗКИ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ ============
def load_test_image(img_id):
    path = f'test/{img_id}.jpg'
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image

# ============ 4. ПРЕДСКАЗАНИЕ ============
print("\nЗагрузка тестовых изображений...")
test_ids = [f.replace('.jpg', '') for f in os.listdir('test') if f.endswith('.jpg')]
print(f"Тестовых изображений: {len(test_ids)}")

def predict_test(model, test_ids, batch_size=32):
    predictions = []
    for i in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[i:i+batch_size]
        batch_images = []
        
        for img_id in batch_ids:
            image = load_test_image(img_id)
            batch_images.append(image)
        
        batch_preds = model.predict(np.array(batch_images), verbose=0)
        predictions.extend(batch_preds)
        
        if (i // batch_size) % 10 == 0:
            print(f"Обработано {min(i+batch_size, len(test_ids))} из {len(test_ids)}...")
    
    return np.array(predictions)

test_predictions = predict_test(model, test_ids)
print(f"Предсказания готовы, форма: {test_predictions.shape}")

# ============ 5. СОЗДАНИЕ SUBMISSION ============
print("\nСоздание submission.csv...")
submission = pd.DataFrame({'id': test_ids})

for i, breed in enumerate(breed_names):
    submission[breed] = test_predictions[:, i]

# Убеждаемся, что порядок колонок правильный
submission = submission[sample_sub.columns]

# Сохраняем
submission.to_csv('submission.csv', index=False)
print("Готово! Файл submission.csv создан.")
print(f"\nПервые 5 строк:")
print(submission.head())

# ============ 6. ПРОВЕРКА ============
print(f"\nПроверка: сумма вероятностей для первого изображения = {test_predictions[0].sum():.4f} (должно быть 1.0)")
predicted_breed = breed_names[np.argmax(test_predictions[0])]
print(f"Предсказанная порода для первого изображения: {predicted_breed}")