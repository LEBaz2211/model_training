import os

def load_list_file(list_file_path):
    """
    Reads a list file where each line has two paths: frame_path and annotation_path.
    """
    pairs = []
    with open(list_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

def ensure_dir(directory):
    """
    Creates the directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
