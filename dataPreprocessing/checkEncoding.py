import os
import chardet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def check_file_encoding(file_path, sample_size=100_000):
    with open(file_path, 'rb') as file:
        raw_data = file.read(sample_size)
    result = chardet.detect(raw_data)
    return result['encoding']

def check_dataset_encodings(directory):
    for root, _, filenames in os.walk(directory):
        for filename in sorted(filenames):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, directory)
                encoding = check_file_encoding(file_path)
                print(f"File: {relative_path}, Encoding: {encoding}")

if __name__ == "__main__":
    check_dataset_encodings(DATA_DIR)
