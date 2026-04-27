from pathlib import Path
from PIL import Image
import json
import uuid
import random 

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def make_uid():
    return "2.25." + str(uuid.uuid4().int)


def get_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size

    return height, width

def random_name():
    first_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    first_name = random.choice(first_names)
    last_name = random.choice(last_names)

    return f"{first_name} {last_name}"

def Sex_options():
    return random.choice(["M", "F", "O"])

def Age_options():
    return random.randint(20, 60)

def study_date():
    year = random.randint(1990, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # To avoid issues with February

    return f"{year:04d}{month:02d}{day:02d}"

def create_metadata(image_path: Path):
    rows, columns = get_image_size(image_path)

    metadata = {
        "PatientName": random_name(),
        "PatientID": "0",
        "PatientAge": Age_options(),
        "PatientSex": Sex_options(),
        "PatientBirthDate": "",
        "StudyDate": study_date(),
        "StudyInstanceUID": make_uid(),
        "SeriesInstanceUID": make_uid(),
        "StudyDescription": "Panoramic Dental X-ray",
        "SeriesDescription": "Demo Series",
        "Modality": "PX",
        "PixelSpacing": [0.08, 0.08],
        "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
        "ImagePositionPatient": [0.0, 0.0, 0.0],
        "SliceThickness": 1.0,
        "Rows": rows,
        "Columns": columns,
        "Manufacturer": "Cybermed Demo"
    }

    return metadata


def generate_json_for_image(image_path: Path, overwrite=False):
    json_path = image_path.with_suffix(".json")

    if json_path.exists() and not overwrite:
        print(f"SKIP: {json_path.name} already exists")
        return

    metadata = create_metadata(image_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"CREATED: {json_path}")


def generate_jsons_in_folder(folder_path: str, overwrite=False):
    folder = Path(folder_path)

    for image_path in folder.iterdir():
        if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            generate_json_for_image(image_path, overwrite=overwrite)


if __name__ == "__main__":
    folder_path = r""  # file path to the folder containing the images
    generate_jsons_in_folder(folder_path, overwrite=False)