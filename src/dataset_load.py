import kagglehub
import shutil
import os

print("Downloading DIV2K...")
path = kagglehub.dataset_download("takihasan/div2k-dataset-for-super-resolution")

print("Path to dataset files:", path)

dst_path = "data/div2k"
os.makedirs(dst_path, exist_ok=True)
print("Copying files...")
shutil.copytree(path, dst_path, dirs_exist_ok=True)

print("Done! Data is now at:", dst_path)