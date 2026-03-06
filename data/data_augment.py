from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x : x / 255.0),
    transforms.Resize(size=(24, 94))
])

test_transforms = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x : x / 255.0),
    transforms.Resize(size=(24, 94))
])