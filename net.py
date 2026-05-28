import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


PATH_TO_TEST_DIR = 'test/'
PATH_TO_SAMPLE_SUB = 'sample_submission.csv'  
OUTPUT_CSV_FILE = 'my_final_submission.csv'  
BATCH_SIZE = 32
DEVICE = torch.device('cpu')
print(f"Устройство: {DEVICE}")

sample_df = pd.read_csv(PATH_TO_SAMPLE_SUB)
breed_columns = list(sample_df.columns[1:])  
num_classes = len(breed_columns)

test_image_paths = glob.glob(os.path.join(PATH_TO_TEST_DIR, '*.jpg')) + \
                   glob.glob(os.path.join(PATH_TO_TEST_DIR, '*.jpeg'))

print(f"Найдено картинок: {len(test_image_paths)}")

class DogTestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return img_id, image

# Уменьшаем картинки до 64x64, как мы и договаривались для простой сети
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

test_dataset = DogTestDataset(image_paths=test_image_paths, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-й слой
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2-й слой
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Финальный слой (32 канала * 16 * 16 пикселей = 8192)
        self.fc = nn.Linear(in_features=8192, out_features=num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1) # Вытягиваем в линию
        x = self.fc(x)
        return x

model = CNN()
model = model.to(DEVICE)
model.eval()


all_ids = []
all_probabilities = []

print("Начинаем обработку фотографий...")
with torch.no_grad():
    for ids, images in tqdm(test_loader):
        images = images.to(DEVICE)

        outputs = model(images)
        
        probabilities = torch.softmax(outputs, dim=1)
        
        probabilities = probabilities.cpu().numpy()
        all_ids.extend(ids)
        all_probabilities.extend(probabilities)

submission_df = pd.DataFrame(all_probabilities, columns=breed_columns)
submission_df.insert(0, 'id', all_ids)

submission_df = submission_df.sort_values(by='id').reset_index(drop=True)
submission_df.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"\nУра! Файл успешно создан: {OUTPUT_CSV_FILE}")
print("Теперь вы можете найти его в той же папке, где запущен ваш код Python, и отправить на сайт!")
