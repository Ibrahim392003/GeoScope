import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import Counter

def generate_class_distribution():
    dataset = datasets.EuroSAT(root="./data", download=False)
    counts = Counter(dataset.targets)
    class_names = dataset.classes

    plt.figure(figsize=(10, 5))
    plt.bar([class_names[i] for i in range(10)], [counts[i] for i in range(10)], color='skyblue')
    plt.xticks(rotation=45)
    plt.title("EuroSAT Class Distribution")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    print("Class distribution chart saved as class_distribution.png")

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

if __name__ == "__main__":
    generate_class_distribution()