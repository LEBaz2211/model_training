#!/usr/bin/env python
import os
import argparse
import yaml
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def find_candidate_paths(raw_dir, rel_path):
    """
    Generate a list of candidate raw image paths.
    It tries:
      - raw_dir/rel_path (as given)
      - If rel_path begins with "train" or "val", remove that and then prepend "train" and "val" to raw_dir.
      - If not, also try inserting "train" and "val" subdirs before rel_path.
    """
    candidates = []
    # Candidate 1: given raw_dir/rel_path.
    candidates.append(os.path.join(raw_dir, rel_path))
    parts = rel_path.split(os.path.sep)
    # If first part is already train/val then remove it and re-add.
    if parts and parts[0] in ["train", "val"]:
        base_rel = os.path.join(*parts[1:])
        candidates.append(os.path.join(raw_dir, "train", base_rel))
        candidates.append(os.path.join(raw_dir, "val", base_rel))
    else:
        candidates.append(os.path.join(raw_dir, "train", rel_path))
        candidates.append(os.path.join(raw_dir, "val", rel_path))
    return candidates

def try_substitutions(path, substitutions):
    """
    For a given path string, try replacing 'labelids_test1' and 'labelids' with each of the substitution strings.
    Returns the new path if the file exists, otherwise returns None.
    """
    for sub in substitutions:
        candidate = path.replace("labelids_test1", sub).replace("labelids", sub)
        print(f"[DEBUG] Trying substitution '{sub}': {candidate}")
        if os.path.exists(candidate):
            return candidate
    return None

def process_file(rel_path, raw_dir, label_dir, output_dir, palette_array):
    """
    For a given label image (located at label_dir/rel_path), this function:
      - Checks if the output composite already exists; if so, skip processing.
      - Loads and colorizes the label image.
      - Attempts to locate the corresponding raw image using candidate paths:
             (a) The given relative path and/or inserted "train" / "val" subdirs.
             (b) Filename substitutions such as ("camera_left", "camera_right", "realsense", "front").
      - Creates a composite image (raw image on the left, colorized label on the right)
        and saves it to output_dir, preserving the folder structure.
    """
    try:
        print(f"\n[DEBUG] Processing label image: {rel_path}")
        full_label_path = os.path.join(label_dir, rel_path)
        output_path = os.path.join(output_dir, rel_path)
        
        # Skip processing if the composite already exists.
        if os.path.exists(output_path):
            print(f"[DEBUG] Skipping {rel_path} because composite already exists.")
            return

        initial_raw_path = os.path.join(raw_dir, rel_path)
        print(f"[DEBUG] Initial raw path: {initial_raw_path}")
        
        # Load and process the label image.
        label_img = Image.open(full_label_path)
        if label_img.mode != "L":
            label_img = label_img.convert("L")
        label_arr = np.array(label_img)
        color_arr = palette_array[label_arr]
        color_img = Image.fromarray(color_arr.astype(np.uint8))
        
        # Attempt to find the raw image using candidate paths.
        candidates = find_candidate_paths(raw_dir, rel_path)
        raw_found = None
        for candidate in candidates:
            print(f"[DEBUG] Trying candidate raw path: {candidate}")
            if os.path.exists(candidate):
                raw_found = candidate
                print(f"[DEBUG] Found raw image at candidate: {raw_found}")
                break
        
        # If no candidate is found, try alternative filename substitutions.
        if not raw_found:
            substitutions = ["windshield_vis","camera_left", "camera_right", "realsense", "front"]
            for candidate in candidates:
                print(f"[DEBUG] No raw image found for candidate: {candidate}")
                sub_candidate = try_substitutions(candidate, substitutions)
                if sub_candidate:
                    raw_found = sub_candidate
                    print(f"[DEBUG] Found raw image with substitution: {raw_found}")
                    break

        if not raw_found:
            raise FileNotFoundError(f"Raw image not found for {rel_path}")
        
        # Load the raw image.
        raw_img = Image.open(raw_found).convert("RGB")
        if raw_img.size != color_img.size:
            print(f"[DEBUG] Resizing color image from {color_img.size} to {raw_img.size}")
            color_img = color_img.resize(raw_img.size, Image.NEAREST)
        
        # Compose the final image.
        width, height = raw_img.size
        composite = Image.new("RGB", (width * 2, height))
        composite.paste(raw_img, (0, 0))
        composite.paste(color_img, (width, 0))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        composite.save(output_path)
        print(f"[DEBUG] Saved composite image: {output_path}")
    except Exception as e:
        print(f"[ERROR] Error processing {rel_path}: {e}")

def process_directory(raw_dir, label_dir, output_dir, palette_array, num_workers):
    """
    Walk through label_dir, gather all image files, and process them in parallel.
    """
    tasks = []
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, label_dir)
                tasks.append(rel_path)
    print(f"[DEBUG] Found {len(tasks)} label images to process.")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, rel_path, raw_dir, label_dir, output_dir, palette_array)
                   for rel_path in tasks]
        for future in futures:
            future.result()

def main():
    parser = argparse.ArgumentParser(
        description="Fast composite: Colorize label images and composite them with raw images (raw on left)."
    )
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Directory containing grayscale label images (e.g. data/rellis/labels).")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory containing raw images (e.g. data/goose-ex/images).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store composite images.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file (to obtain the palette).")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8).")
    args = parser.parse_args()

    # Load the configuration and retrieve the palette.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if "superclasses" in config and "palette" in config["superclasses"]:
        palette = config["superclasses"]["palette"]
    elif "classes" in config and "palette" in config["classes"]:
        palette = config["classes"]["palette"]
    else:
        raise ValueError("No palette found in config file.")
    palette_array = np.array(palette, dtype=np.uint8)
    print(f"[DEBUG] Using palette with shape: {palette_array.shape}")
    
    process_directory(args.raw_dir, args.label_dir, args.output_dir, palette_array, args.num_workers)

if __name__ == "__main__":
    main()