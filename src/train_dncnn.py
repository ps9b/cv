import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dncnn_model import DnCNN


class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir

        self.files = os.listdir(clean_dir)

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        clean = cv2.imread(os.path.join(self.clean_dir, filename))
        noisy = cv2.imread(os.path.join(self.noisy_dir, filename))

        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

        clean = self.transform(clean)
        noisy = self.transform(noisy)

        return noisy, clean


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DenoisingDataset(
        '../datasets/train',
        '../noisy_data/poisson'
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DnCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for noisy, clean in tqdm(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}')

    torch.save(model.state_dict(), '../models/dncnn.pth')


if __name__ == '__main__':
    train()