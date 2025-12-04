import numpy as np
from utils import saveArrays, loadArrays
from pathlib import Path
from matplotlib import pyplot as plt

script_dir = Path(__file__).parent 
restnet_file_name = script_dir.joinpath("..", "plots", "resnet_arrays.npz")
swinir_file_name = script_dir.joinpath("..", "plots", "swinir_arrays.npz")


resnet_train_loss, resnet_valid_loss = loadArrays(restnet_file_name, "arrayA", "arrayB")
swinir_train_loss, swinir_valid_loss = loadArrays(swinir_file_name, "arrayA", "arrayB")
num_epochs = 25

plt.figure(1)
plt.plot([i for i in range(1, num_epochs + 1)], resnet_train_loss, label="Training Loss")
plt.plot([i for i in range(1, num_epochs + 1)], resnet_valid_loss, label="Validation Loss")
plt.title("ResNet Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 0.02)
plt.legend()


plt.figure(2)
plt.plot([i for i in range(1, num_epochs + 1)], swinir_train_loss, label="Training Loss")
plt.plot([i for i in range(1, num_epochs + 1)], swinir_valid_loss, label="Validation Loss")
plt.title("SWINIR Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0, 0.02)
plt.legend()


plt.show()






