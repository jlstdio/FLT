import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.cuda import is_available
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
from model.testModel_wo_softmax import testNN
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

data_dir = '../dataset/cifar10'
cifar_dataloader = cifar10Dataloader(data_dir)
(x_train, y_train), (x_test, y_test) = cifar_dataloader.load_data()
classes = list(set(y_test))

rootModelPath = './data'
testName = 'm3 - case 1 - 3'
clipN = 5
isBCE = True
rootModelPath = f'{rootModelPath}/{testName}/rootModel-{testName}.pth'

x_test = np.array(x_test)
y_test = np.array(y_test)
if isBCE:
    y_test = np.eye(10)[y_test]  # BCE

X_validation = torch.tensor(x_test, dtype=torch.float32).permute(0, 3, 1, 2)
if isBCE:
    y_validation = torch.tensor(y_test, dtype=torch.float32)  # BCE
else:
    y_validation = torch.tensor(y_test, dtype=torch.long)  # CE

X_validation = F.normalize(X_validation, dim=0)
validation_dataset = TensorDataset(X_validation, y_validation)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

model = testNN()

cudaId = 0
device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")
model_state_dict = torch.load(rootModelPath, map_location=device)
model.load_state_dict(model_state_dict)
model = model.to(device)

model.eval()
features = []
labels = []
with torch.no_grad():
    # for images, targets in testloader:
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        features.append(outputs)
        if isBCE:
            targets = targets.to(device)  # BCE
        else:
            targets = targets.long().to(device)  # CE
        labels.append(targets)

features = torch.cat(features).cpu().numpy()
labels = torch.cat(labels).cpu().numpy()

# Convert one-hot labels to class indices
if isBCE:
    labels = np.argmax(labels, axis=1)  # BCE

# 4. t-SNE 적용
tsne = TSNE(n_components=2, random_state=seed)
features_tsne = tsne.fit_transform(features)

# 5. 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
if isBCE:
    plt.title(f't-SNE | {testName} | BCE | clipping N = {clipN}')
else:
    plt.title(f't-SNE | {testName} | CE | no clipping')
plt.show()
