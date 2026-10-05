from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


DATASET_PATH = Path(__file__).with_name("CNN_Trans")
MODEL_PATH = Path(__file__).with_name("shape_cnn.pt")
IMAGE_SIZE = 224
CLASS_NAMES = ("球", "三棱柱", "圆柱", "圆锥", "长方体", "正方体")
MODEL_VERSION = 7
CPU = torch.device("cpu")
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ShapeCNN(nn.Module):
    def __init__(self, class_count=len(CLASS_NAMES)):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad = True
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone.fc.in_features, class_count),
        )

    def forward(self, images):
        return self.backbone(images)


def _read_image(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return _crop_object(image)


def _object_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated = hsv[:, :, 1] >= 45
    bright = hsv[:, :, 2] >= 65
    non_wood_hue = (hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 27)
    yellow = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 70)
    mask = np.where(saturated & bright & (non_wood_hue | yellow), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _crop_object(image):
    mask = _object_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * 0.02:
        return image
    x, y, width, height = cv2.boundingRect(contour)
    padding = max(width, height) // 10
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image.shape[1], x + width + padding)
    y1 = min(image.shape[0], y + height + padding)
    return image[y0:y1, x0:x1]


def _prepare_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
    mean = torch.from_numpy(IMAGE_MEAN).view(3, 1, 1)
    std = torch.from_numpy(IMAGE_STD).view(3, 1, 1)
    return (tensor - mean) / std


def _augment(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    angle = random.uniform(-12, 12)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, random.uniform(0.9, 1.1))
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


class ShapeImageDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = _read_image(path)
        if self.augment:
            image = _augment(image)
        return _prepare_tensor(image), label


def split_dataset(dataset_path=DATASET_PATH, seed=7):
    random_generator = random.Random(seed)
    train_samples = []
    validation = []
    test = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_files = sorted(
            path for path in (dataset_path / class_name).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not class_files:
            raise FileNotFoundError(f"类别目录为空或不存在: {dataset_path / class_name}")
        random_generator.shuffle(class_files)
        total = len(class_files)
        train_end = max(1, int(total * 0.7))
        val_end = min(total - 1, train_end + max(1, int(total * 0.15)))
        train_samples.extend((path, label) for path in class_files[:train_end])
        validation.extend((path, label) for path in class_files[train_end:val_end])
        test.extend((path, label) for path in class_files[val_end:])

    random_generator.shuffle(train_samples)
    random_generator.shuffle(validation)
    random_generator.shuffle(test)
    return train_samples, validation, test


def _accuracy(model, loader):
    correct = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            count += labels.numel()
    return correct / count if count else 0.0


def train_model(model_path=MODEL_PATH, dataset_path=DATASET_PATH, epochs=60):
    torch.manual_seed(7)
    random.seed(7)
    train_samples, val_samples, test_samples = split_dataset(dataset_path)
    print(f"数据集: 训练 {len(train_samples)} 张, 验证 {len(val_samples)} 张, 测试 {len(test_samples)} 张")
    train_loader = DataLoader(ShapeImageDataset(train_samples, augment=True), batch_size=16, shuffle=True)
    val_loader = DataLoader(ShapeImageDataset(val_samples), batch_size=16)
    test_loader = DataLoader(ShapeImageDataset(test_samples), batch_size=16)

    model = ShapeCNN().to(CPU)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4, weight_decay=1e-4)
    train_counts = torch.bincount(torch.tensor([label for _, label in train_samples]), minlength=len(CLASS_NAMES)).float()
    class_weights = torch.sqrt(train_counts.sum() / train_counts.clamp_min(1))
    class_weights /= class_weights.mean()
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    best_state = None
    best_accuracy = -1.0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()

        validation_accuracy = _accuracy(model, val_loader)
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        print(f"第 {epoch + 1:02d}/{epochs} 轮: 验证集准确率 {validation_accuracy:.1%}")

    if best_state is None:
        raise RuntimeError("训练未生成有效模型权重。")
    model.load_state_dict(best_state)
    test_accuracy = _accuracy(model, test_loader)
    torch.save({"state_dict": model.state_dict(), "class_names": CLASS_NAMES, "version": MODEL_VERSION}, model_path)
    print(f"测试集准确率: {test_accuracy:.1%}")
    print(f"模型已保存: {model_path}")
    return model


def load_model(model_path=MODEL_PATH):
    model = ShapeCNN().to(CPU)
    if not model_path.exists():
        return train_model(model_path)
    checkpoint = torch.load(model_path, map_location=CPU, weights_only=True)
    if checkpoint.get("class_names") != CLASS_NAMES or checkpoint.get("version") != MODEL_VERSION:
        print("检测到旧预处理模型，重新训练六类物块模型...")
        return train_model(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _prepare_image(image):
    if image.ndim == 3:
        image = _crop_object(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return _prepare_tensor(image)[None, :, :, :]


def predict_image(model, image):
    """Predict from a cropped BGR or grayscale object image."""
    with torch.inference_mode():
        probabilities = torch.softmax(model(_prepare_image(image)), dim=1)[0]
    confidence, class_index = torch.max(probabilities, dim=0)
    return CLASS_NAMES[int(class_index)], float(confidence)


def predict_shape(model, contour, frame):
    """Locate a contour in frame, crop it, and classify the original object image."""
    x, y, width, height = cv2.boundingRect(contour)
    padding = max(width, height) // 8
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(frame.shape[1], x + width + padding), min(frame.shape[0], y + height + padding)
    return predict_image(model, frame[y0:y1, x0:x1])
