import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import timm
from PIL import Image

class CustomHierarchicalImageDataset(Dataset):
    def __init__(self, samples, transform=None, class_to_idx=None, classes=None):
        self.samples = samples
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.classes = classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# -------- Config ----------
feedback_dir = "retrain/feedback_data"
original_model_path = "classify_models/efficientnet_b0_best.pth"
temp_new_model_path = "retrain/tmp_model.pth"
best_models_dir = "retrain/retrained_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 3
batch_size = 16


# -------- Load feedback samples ----------
def load_feedback_samples(feedback_dir):
    samples = []
    class_names = sorted(os.listdir(feedback_dir))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    for cls in class_names:
        class_path = os.path.join(feedback_dir, cls)
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append((os.path.join(class_path, fname), class_to_idx[cls]))
    return samples, class_to_idx, class_names


# -------- Load EfficientNet model ----------
def load_model(num_classes, path):
    model = timm.create_model("efficientnet_b0", pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# -------- Train model on feedback ----------
def train_model_on_feedback(model, train_loader, val_loader):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate_model(model, val_loader)
        print(f"[Epoch {epoch + 1}] Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), temp_new_model_path)
    return best_acc


# -------- Evaluate model ----------
def evaluate_model(model, val_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            preds += outputs.argmax(1).cpu().tolist()
            labels += y.tolist()
    return accuracy_score(labels, preds)


# -------- Lấy đường dẫn model retrain kế tiếp ----------
def get_next_model_path(save_dir=best_models_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.startswith("model_retrain_") and f.endswith(".pth")]
    nums = [int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(save_dir, f"model_retrain_{next_num:03d}.pth")


# -------- Lấy model tốt nhất hiện tại ----------
def get_best_model_path(models_dir=best_models_dir):
    os.makedirs(models_dir, exist_ok=True)
    models = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not models:
        return None
    models = sorted(models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(models_dir, models[-1])


# -------- MAIN --------
if __name__ == "__main__":
    print("[INFO] Loading feedback data...")
    samples, class_to_idx, class_names = load_feedback_samples(feedback_dir)

    if len(samples) < 10:
        print("[WARNING] Not enough feedback samples (<10). Skipping retrain.")
        exit()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_s, val_s = train_test_split(samples, test_size=0.2, stratify=[s[1] for s in samples])
    train_ds = CustomHierarchicalImageDataset(train_s, transform, class_to_idx, class_names)
    val_ds = CustomHierarchicalImageDataset(val_s, transform, class_to_idx, class_names)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Tải mô hình tốt nhất hiện tại (nếu có), không thì dùng model gốc
    best_model_path = get_best_model_path()
    if best_model_path:
        print(f"[INFO] Found best model: {best_model_path}")
        old_model = load_model(len(class_names), best_model_path)
    else:
        print("[INFO] No previous model found. Using original model.")
        old_model = load_model(len(class_names), original_model_path)

    # Huấn luyện model mới từ original model
    print("[INFO] Training model on feedback data...")
    current_model = load_model(len(class_names), original_model_path)
    train_model_on_feedback(current_model, train_loader, val_loader)

    # So sánh và lưu nếu tốt hơn
    new_model = load_model(len(class_names), temp_new_model_path)

    print("[INFO] Evaluating old and new models...")
    old_acc = evaluate_model(old_model, val_loader)
    new_acc = evaluate_model(new_model, val_loader)

    print(f"[RESULT] Old Acc: {old_acc:.4f} | New Acc: {new_acc:.4f}")
    if new_acc > old_acc:
        new_model_path = get_next_model_path()
        torch.save(new_model.state_dict(), new_model_path)
        print(f"New model better. Saved to {new_model_path}")
    else:
        print("New model not better. Keeping current best model.")


#python retrain/retrain.py

