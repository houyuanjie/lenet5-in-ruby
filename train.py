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


# 训练并保存模型

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
print("-" * 80)


# 生成 model/setup.rb 脚本

setup_rb_path = Path(__file__).parent / "model" / "setup.rb"


def format_array_as_ruby(array, fetch_fn=float):
    length = len(array)
    ruby_str = "["
    for i in range(length):
        value = array[i]
        ruby_str += str(fetch_fn(value)) + ","
    ruby_str += "]"
    return ruby_str


def format_matrix_as_ruby(matrix, fetch_fn=float):
    matrix_height, matrix_width = matrix.shape
    ruby_str = "Matrix["
    for row in range(matrix_height):
        ruby_str += "["
        for col in range(matrix_width):
            value = matrix[row, col]
            ruby_str += str(fetch_fn(value)) + ","
        ruby_str += "],"
    ruby_str += "]"
    return ruby_str


def format_conv_weights_as_ruby(conv_weights, fetch_fn=float):
    out_channels, in_channels, kernel_height, kernel_width = conv_weights.shape
    ruby_str = "["
    for out_chn in range(out_channels):
        ruby_str += "["
        for in_chn in range(in_channels):
            ruby_str += "Matrix["
            for row in range(kernel_height):
                ruby_str += "["
                for col in range(kernel_width):
                    value = conv_weights[out_chn, in_chn, row, col]
                    ruby_str += str(fetch_fn(value)) + ","
                ruby_str += "],"
            ruby_str += "],"
        ruby_str += "],"
    ruby_str += "]"
    return ruby_str


conv1_weights = model.conv1.weight.data.cpu().numpy()
conv1_weights_ruby = format_conv_weights_as_ruby(conv1_weights)

conv1_bias = model.conv1.bias.data.cpu().numpy()
conv1_bias_ruby = format_array_as_ruby(conv1_bias)

conv2_weights = model.conv2.weight.data.cpu().numpy()
conv2_weights_ruby = format_conv_weights_as_ruby(conv2_weights)

conv2_bias = model.conv2.bias.data.cpu().numpy()
conv2_bias_ruby = format_array_as_ruby(conv2_bias)

fc1_weight = model.fc1.weight.data.cpu().numpy()
fc1_weight_ruby = format_matrix_as_ruby(fc1_weight)

fc1_bias = model.fc1.bias.data.cpu().numpy()
fc1_bias_ruby = format_array_as_ruby(fc1_bias)

fc2_weight = model.fc2.weight.data.cpu().numpy()
fc2_weight_ruby = format_matrix_as_ruby(fc2_weight)

fc2_bias = model.fc2.bias.data.cpu().numpy()
fc2_bias_ruby = format_array_as_ruby(fc2_bias)

fc3_weight = model.fc3.weight.data.cpu().numpy()
fc3_weight_ruby = format_matrix_as_ruby(fc3_weight)

fc3_bias = model.fc3.bias.data.cpu().numpy()
fc3_bias_ruby = format_array_as_ruby(fc3_bias)


def make_setup_rb(file):
    file.write("require 'matrix'\n")
    file.write("\n")
    file.write("module Nn\n")
    file.write("  class LeNet5\n")
    file.write("    def self.setup(instance)\n")
    file.write("      instance.conv1.weights = {}.freeze\n".format(conv1_weights_ruby))
    file.write("      instance.conv1.bias = {}.freeze\n".format(conv1_bias_ruby))
    file.write("      instance.conv2.weights = {}.freeze\n".format(conv2_weights_ruby))
    file.write("      instance.conv2.bias = {}.freeze\n".format(conv2_bias_ruby))
    file.write("      instance.fc1.weight = {}.freeze\n".format(fc1_weight_ruby))
    file.write("      instance.fc1.bias = {}.freeze\n".format(fc1_bias_ruby))
    file.write("      instance.fc2.weight = {}.freeze\n".format(fc2_weight_ruby))
    file.write("      instance.fc2.bias = {}.freeze\n".format(fc2_bias_ruby))
    file.write("      instance.fc3.weight = {}.freeze\n".format(fc3_weight_ruby))
    file.write("      instance.fc3.bias = {}.freeze\n".format(fc3_bias_ruby))
    file.write("    end\n")
    file.write("  end\n")
    file.write("end\n")


with open(setup_rb_path, "w") as file:
    file.write("# This file is generated by train.py.\n")
    file.write("# Do not edit manually.\n")
    make_setup_rb(file)

print("setup.rb generated.")
