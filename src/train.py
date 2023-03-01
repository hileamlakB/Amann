import torch
import torch.nn as nn
import torch.optim as optim
import amann as amn
from torch.utils.data import DataLoader
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

# open dataset, train the model, and save the model

data_dir = "../dataset/"
dataset = amn.AmharicDataset(data_dir)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=80, shuffle=True)


model = amn.Amann()
criterion = nn.CrossEntropyLoss()
optimzer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimzer.zero_grad()

        outputs = model(inputs)
        # print("input shape", inputs.shape, outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimzer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# pandas dataframe to load the csv map file
df = pd.read_csv("../supported_chars.csv")
prop = FontProperties()
prop.set_file("../Fonts/NotoSerif.ttf")


# test model accuracy
model.eval()
correct = 0
total = 0

# Iterate over test dataset
with torch.no_grad():
    for inputs, labels in test_loader:
        # Pass input through model to get predictions
        outputs = model(inputs)

        # Get predicted labels
        _, predicted = torch.max(outputs.data, 1)

        # Update total count and correct count
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Visualize input with image viewer along with the prediction
        for i in range(inputs.size(0)):
            image = inputs[i].numpy()
            is_correct = predicted[i].item() == labels[i].item()

            label_str = "Correct" if is_correct else "Mistaken"
            character = df["Character"][labels[i].item()]
            predicted_char = df["Character"][predicted[i].item()]

            # Save test results
            plt.imshow(np.squeeze(image), cmap='gray')
            plt.title(
                f"{label_str}: Actual -> {character}, predicted -> {predicted_char}", fontproperties=prop)
            plt.savefig(f"../test_results/{label_str}_{i}_{correct}.png")

# Calculate accuracy
accuracy = 100 * correct / total

# Print accuracy
print('Accuracy on test set: {:.2f}%'.format(accuracy))

# save model to disk
torch.save(model.state_dict(), "amann.pt")
