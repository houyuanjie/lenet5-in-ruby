import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from pathlib import Path
from model.le_net5 import LeNet5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LeNet5().to(device)

dataset_dir = Path(__file__).parent / "dataset"
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=dataset_dir, train=True, transform=transform)
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root=dataset_dir, train=False, transform=transform)
test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def optimize(model, dataloader, loss_fn, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")


def test(model, dataloader, loss_fn):
    model.eval()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(dataloader)
    accuracy = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy


epochs = 50
best_loss = float("inf")
best_accuracy = 0.0
best_model_save_path = Path(__file__).parent / "model" / "best_model.pt"

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    optimize(model, train_dataloader, loss_fn, optimizer)
    test_loss, accuracy = test(model, test_dataloader, loss_fn)
    scheduler.step()

    if test_loss < best_loss:
        best_loss = test_loss
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_save_path)
        print(
            "Saved best_model.pt with:\n"
            + f"    Loss: {best_loss:.4f}\n"
            + f"    Accuracy: {best_accuracy:.2f}%"
        )

    print("-" * 80)

print("Training complete.")
print(f"Best Loss: {best_loss:.4f}.")
print(f"Best Accuracy: {best_accuracy:.2f}%")
print(f"Best Model Saved at: {best_model_save_path}")
