import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import Counter

def generate_class_distribution():
    dataset = datasets.EuroSAT(root="./data", download=False)
    counts = Counter(dataset.targets) ## counts how many images belong to each class
    class_names = dataset.classes

    plt.figure(figsize=(10, 5))
    plt.bar([class_names[i] for i in range(10)], [counts[i] for i in range(10)], color='skyblue')
    plt.xticks(rotation=45)
    plt.title("EuroSAT Class Distribution")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    print("Class distribution chart saved as class_distribution.png") 

    ## we can see that the imbalance ratio between 2 classes is 1.5, which is not a big deal. 
    ## So we can use the dataset as it is without any oversampling or undersampling.

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)), ## converting from 64x64 to 224x224 because the model(torchvision)
                                       ## is pretrained on ImageNet(size 224x224)

        transforms.ToTensor(), ## normal in images are stored as numbers from 0 to 255, but we need to 
                               ## convert them to numbers from 0 to 1 for the model to work properly.

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) #some images are brighter than others, so we need to normalize them to have the same brightness.
          
    ])

import torchvision.datasets as datasets

def print_dataset_metadata():
    dataset = datasets.EuroSAT(root="./data", download=False)
    sample_img, label = dataset[0]
    print(f"Sample Image Type: {type(sample_img)}")
    print(f"Original Image Size: {sample_img.size}") # (64, 64)
    print(f"Number of Channels: {len(sample_img.getbands())}") # 3 (RGB)

if __name__ == "__main__":
    print_dataset_metadata()

if __name__ == "__main__":
    generate_class_distribution()