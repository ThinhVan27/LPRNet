import cv2
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import os

def sharpening(img):
    if not isinstance(img, np.ndarray):
        img =  np.array(img, dtype=np.float32)
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    gaussian_blur = cv2.GaussianBlur(img, (7, 7), 3.0)
    edges = cv2.subtract(img, gaussian_blur)
    sharpened = cv2.add(img, edges)
    sharpened = np.clip(img, 0, 1)
    return torch.tensor(sharpened, dtype=torch.float32).permute(2, 0, 1)

# Train transforms khuyến nghị
train_transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),  # về [0,1]
    transforms.Resize((40, 100), antialias=True),
    transforms.Lambda(sharpening), 
    transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.9, 1.1), shear=5),
])

# Test transforms: nhẹ hơn, không random mạnh
test_transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((40, 100), antialias=True),
    transforms.Lambda(sharpening),
    
])

if __name__ == "__main__":
    # Visualize some samples
    track1 = "train/Scenario-A/Brazilian/track_00001"
    track2 = "train/Scenario-A/Mercosur/track_02489"
    track3 = "train/Scenario-B/Brazilian/track_10001"
    track4 = "train/Scenario-B/Mercosur/track_12602"
    
    samples = [track1, track2, track3, track4]
    
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 7))
    
    for i, ax in enumerate(axes):
        for j in range(5):
            img = Image.open(os.path.join(samples[i], f"hr-00{j+1}.jpg" if f"hr-00{j+1}.jpg" in os.listdir(samples[i]) else f"hr-00{j+1}.png"))
            train_img = train_transforms(img)
            test_img = test_transforms(img)
            ax[j].set_axis_off()
            ax[j].set_title(f"{samples[i]}/hr-00{j+1}", fontdict={'fontsize': 8})
            ax[j].imshow(train_img.permute(1, 2, 0))
            
    plt.tight_layout()
    plt.savefig("augmented_samples.png")
