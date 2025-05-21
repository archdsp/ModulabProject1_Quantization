from voc import VOCDataset, VOC_CLASSES
from detector import SimpleObjectDetector
from loss import detection_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


def train(device="mps"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = VOCDataset("VOCdevkit/VOC2007", transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = SimpleObjectDetector(num_classes=len(VOC_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for images, targets in tqdm(loader):
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)  # [B, S, S, 4+C]
            loss = detection_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()