import os
import hashlib
import shutil

CURRENT_DATASET = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai\data\train"

KAGGLE_DATASET = r"C:\Users\vedan\.cache\kagglehub\datasets\mohamedhanyyy\chest-ctscan-images\versions\1"

CLASS_MAP = {
    "adenocarcinoma": "adenocarcinoma",
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "adenocarcinoma",
    "large.cell.carcinoma": "large cell carcinoma",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "large cell carcinoma",
    "squamous.cell.carcinoma": "squamous cell carcinoma",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "squamous cell carcinoma",
    "normal": "Normal cases"
}

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

def md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():

    print("=" * 60)
    print("ADDING NEW NON-DUPLICATE DATA")
    print("=" * 60)

    existing_hashes = set()

    print("\nChecking existing dataset...")

    for root, dirs, files in os.walk(CURRENT_DATASET):
        for file in files:
            if file.lower().endswith(EXTENSIONS):
                path = os.path.join(root, file)
                existing_hashes.add(md5(path))

    print(f"Existing unique images: {len(existing_hashes)}")

    added = {
        "adenocarcinoma": 0,
        "large cell carcinoma": 0,
        "squamous cell carcinoma": 0,
        "Normal cases": 0
    }

    skipped = 0

    print("\nSearching downloaded Kaggle dataset...")

    for root, dirs, files in os.walk(KAGGLE_DATASET):

        folder_name = os.path.basename(root)

        if folder_name not in CLASS_MAP:
            continue

        target_class = CLASS_MAP[folder_name]
        target_dir = os.path.join(CURRENT_DATASET, target_class)

        os.makedirs(target_dir, exist_ok=True)

        for file in files:

            if not file.lower().endswith(EXTENSIONS):
                continue

            source_path = os.path.join(root, file)

            file_hash = md5(source_path)

            if file_hash in existing_hashes:
                skipped += 1
                continue

            base, ext = os.path.splitext(file)

            new_name = f"kaggle_new_{added[target_class] + 1}_{file}"

            destination = os.path.join(target_dir, new_name)

            counter = 1

            while os.path.exists(destination):
                new_name = f"kaggle_new_{added[target_class] + 1}_{counter}{ext}"
                destination = os.path.join(target_dir, new_name)
                counter += 1

            shutil.copy2(source_path, destination)

            existing_hashes.add(file_hash)
            added[target_class] += 1

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    for cls, count in added.items():
        print(f"{cls}: {count} new images")

    print("-" * 60)
    print(f"TOTAL NEW IMAGES ADDED: {sum(added.values())}")
    print(f"DUPLICATES SKIPPED: {skipped}")
    print("=" * 60)

if __name__ == "__main__":
    main()