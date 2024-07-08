import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class modelManager:
    def __init__(self, modelPath=None, model=None, device=None, enableParallel=True, lr=0.0001):
        self.model = model
        self.device = device
        seed = 1234
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        if not torch.cuda.is_available():
            self.model.to('cpu')

        if modelPath is not None:
            prefixRemoved = remove_module_prefix(torch.load(modelPath, map_location=torch.device('cpu')))
            self.model.load_state_dict(prefixRemoved)

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1 and enableParallel:
           self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_model(self, train_loader, val_loader, num_epochs):

        f = open("trainEvolution.txt", "w")
        f.write('starting seq \n')
        f.close()

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for acc_data, audio_data, labels in train_loader:
                acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(
                    self.device)

                self.optimizer.zero_grad()
                outputs = self.model(acc_data, audio_data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for acc_data, audio_data, labels in val_loader:
                    acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(
                        self.device)

                    outputs = self.model(acc_data, audio_data)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            f = open("trainEvolution.txt", "a")
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
            f.close()

            wandb.log({"Training loss": train_loss})
            wandb.log({"Validation loss": val_loss})

            torch.save(self.model.state_dict(), f'pths/gesture_transformer_epoch{epoch}.pth')

    def print_model(self):
        return self.model

    def test_model(self, test_loader, matrix=False):
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for acc_data, audio_data, labels in test_loader:
                acc_data, audio_data, labels = acc_data.to(self.device), audio_data.to(self.device), labels.to(
                    self.device)

                outputs = self.model(acc_data, audio_data)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        test_loss /= len(test_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Test Loss: {test_loss:.4f}, F1 Score: {f1:.4f}")

        if matrix is True:
            # Generate Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plotting
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=np.unique(all_labels))
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            plt.title('Normalized Confusion Matrix')
            plt.savefig('confusion_matrix.png')

        return f1
