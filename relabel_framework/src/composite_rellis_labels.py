#!/usr/bin/env python
import os
import argparse
import yaml
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import raw_to_seq from relabeler to convert raw labels if needed.
from relabeler import raw_to_seq

def load_list_file(list_file_path):
    """
    Reads a list file where each line contains two paths:
    one for the raw image and one for the corresponding label.
    """
    pairs = []
    with open(list_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

def colorize_label(label_arr, palette_array):
    """
    Uses vectorized indexing to convert a 2D label array into an RGB image
    by replacing each pixel with the corresponding color from palette_array.
    """
    return palette_array[label_arr]

def process_pair(frame_path, label_path, root_dir, palette_array, id_seq, output_base):
    """
    For a given pair (raw image and label), load the raw image and grayscale label,
    optionally remap the label values using id_seq (if provided),
    colorize the label using palette_array, composite them side-by-side,
    and save the composite image in output_base.
    """
    # Build full paths
    raw_full = os.path.join(root_dir, frame_path)
    label_full = os.path.join(root_dir, label_path)
    
    try:
        raw_img = Image.open(raw_full).convert("RGB")
    except Exception as e:
        print(f"Error loading raw image {raw_full}: {e}")
        return None
    
    try:
        label_img = Image.open(label_full)
        if label_img.mode != "L":
            label_img = label_img.convert("L")
    except Exception as e:
        print(f"Error loading label image {label_full}: {e}")
        return None

    # Convert label image to numpy array.
    label_arr = np.array(label_img)
    # If an id_seq mapping is provided, convert raw label values to sequential indices.
    if id_seq is not None:
        label_arr = raw_to_seq(label_arr, id_seq)
    
    # Colorize label using vectorized indexing
    color_label_arr = colorize_label(label_arr, palette_array)
    color_label_img = Image.fromarray(color_label_arr.astype(np.uint8))
    
    # Ensure both images have the same dimensions; if not, resize the colorized label.
    if raw_img.size != color_label_img.size:
        color_label_img = color_label_img.resize(raw_img.size, Image.NEAREST)
    
    # Create composite image: raw image on left, colorized label on right.
    width, height = raw_img.size
    composite = Image.new("RGB", (width * 2, height))
    composite.paste(raw_img, (0, 0))
    composite.paste(color_label_img, (width, 0))
    
    # Construct output path: preserve label image's directory structure.
    rel_dir = os.path.dirname(label_path)
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    out_dir = os.path.join(output_base, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, base_name + "_composite.png")
    
    composite.save(output_path)
    print(f"Saved composite: {output_path}")
    return output_path

def process_all(root_dir, list_files, palette_array, id_seq, output_base, num_workers):
    """
    For each list file (e.g. train, val, test), load the pairs (raw image, label)
    and process them in parallel using a ProcessPoolExecutor.
    """
    all_pairs = []
    for key, list_file in list_files.items():
        list_path = os.path.join(root_dir, list_file)
        pairs = load_list_file(list_path)
        all_pairs.extend(pairs)
    print(f"Processing {len(all_pairs)} image pairs...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_pair, frame_path, label_path, root_dir, palette_array, id_seq, output_base):
            (frame_path, label_path)
            for frame_path, label_path in all_pairs
        }
        for future in as_completed(futures):
            pair = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Error processing pair {pair}: {exc}")

def main():
    parser = argparse.ArgumentParser(
        description="Composite Rellis: Colorize original labels and combine with raw images (raw on left)."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file for Rellis (e.g., config/rellis_test_1.yaml).")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory for composite images.")
    args = parser.parse_args()

    # Load configuration.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_config = config["dataset"]
    root_dir = dataset_config["root_dir"]
    list_files = dataset_config["list_files"]
    
    # Use the base palette from the 'classes' section to colorize the original labels.
    if "classes" in config and "palette" in config["classes"]:
        palette = config["classes"]["palette"]
    else:
        raise ValueError("No palette found in the config file under 'classes'.")
    
    palette_array = np.array(palette, dtype=np.uint8)
    print(f"Using palette with shape: {palette_array.shape}")
    
    # Optionally, load the id_seq mapping for converting raw label values.
    id_seq = config["classes"].get("id_seq", None)
    
    # Process all image pairs in parallel.
    process_all(root_dir, list_files, palette_array, id_seq, args.output_dir, args.num_workers)

if __name__ == "__main__":
    main()
