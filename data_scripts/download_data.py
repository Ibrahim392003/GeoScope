
import json
import platform
from datetime import datetime, timezone

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from common import DATA_DIR, REPORT_DIR, ensure_dirs


def download_eurosat():
    ensure_dirs()
    print("Downloading EuroSAT dataset...")
    dataset = datasets.EuroSAT(
        root=str(DATA_DIR),
        download=True,
        transform=transforms.ToTensor(),
    )
    print("Dataset successfully downloaded!")
    print(f"Total satellite images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    log = {
        "dataset": "EuroSAT (RGB, Sentinel-2)",
        "reference": "Helber et al., IEEE JSTARS 2019; https://github.com/phelber/EuroSAT",
        "downloaded_via": f"torchvision {torchvision.__version__}  datasets.EuroSAT(download=True)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "num_images": len(dataset),
        "classes": dataset.classes,
    }
    (REPORT_DIR / "download_log.json").write_text(json.dumps(log, indent=2))
    print("Download log written to report/download_log.json")


if __name__ == "__main__":
    download_eurosat()
