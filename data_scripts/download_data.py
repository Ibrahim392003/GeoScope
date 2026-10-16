import torchvision.datasets as datasets
import torchvision.transforms as transforms

def download_eurosat():
    data_dir = "./data"
    print("Downloading EuroSAT dataset...")
    dataset = datasets.EuroSAT(
        root=data_dir, 
        download=True, 
        transform=transforms.ToTensor()
    )
    print("Dataset successfully downloaded!")
    print(f"Total satellite images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

if __name__ == "__main__":
    download_eurosat()