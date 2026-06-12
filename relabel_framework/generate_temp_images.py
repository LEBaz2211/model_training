#!/usr/bin/env python
import os
import subprocess
import argparse
import difflib
import re
import yaml

def process_file_list(file_list_path):
    """Read the file list (one file name per line) and return the list."""
    with open(file_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def extract_frame_number(filename):
    """
    Extracts the frame number from a filename using the pattern __<digits>_
    For example, from:
      2022-08-30_siegertsbrunn_feldwege__0470_1661859255430126959_color.png
    it extracts "0470".
    """
    match = re.search(r'__([0-9]+)_', filename)
    return match.group(1) if match else None

def construct_paths(filename):
    """
    Given a filename like:
      2022-07-07_campus_no_ptp__0001_1657197246879010831_labelids.png
    extract the scene (folder name) and build the relative paths:
      sample_image (annotation): labels/train/<scene>/<filename with "labelids" replaced by "color">
      frame_image: images/train/<scene>/<filename with "labelids" replaced by "windshield_vis">
    """
    if '__' not in filename:
        raise ValueError(f"Filename '{filename}' does not contain '__' to separate the scene.")
    
    scene, _ = filename.split('__', 1)
    # Replace suffixes for annotation and frame
    sample_image_file = filename.replace("labelids", "color")
    frame_image_file = filename.replace("labelids", "windshield_vis")
    
    sample_image_rel = os.path.join("labels", "train", scene, sample_image_file)
    frame_image_rel = os.path.join("images", "train", scene, frame_image_file)
    return sample_image_rel, frame_image_rel

def find_best_match_with_frame(root_dir, relative_path, expected_frame):
    """
    Check if the file at root_dir/relative_path exists.
    If not, list the directory and filter candidates to only those containing expected_frame.
    Then use difflib to find the closest filename among these candidates.
    Returns a new relative path (relative to root_dir) if a close match is found.
    """
    full_path = os.path.join(root_dir, relative_path)
    if os.path.exists(full_path):
        return relative_path
    else:
        dir_name = os.path.dirname(full_path)
        expected_file = os.path.basename(full_path)
        try:
            candidates = os.listdir(dir_name)
        except Exception as e:
            print(f"Directory {dir_name} does not exist: {e}")
            return relative_path
        
        # Filter candidates that contain the expected frame number.
        filtered = [cand for cand in candidates if expected_frame in cand]
        if filtered:
            matches = difflib.get_close_matches(expected_file, filtered, n=1, cutoff=0.6)
        else:
            matches = difflib.get_close_matches(expected_file, candidates, n=1, cutoff=0.6)
        if matches:
            best = matches[0]
            new_full_path = os.path.join(dir_name, best)
            new_relative = os.path.relpath(new_full_path, root_dir)
            print(f"Replacing '{relative_path}' with closest match '{new_relative}'")
            return new_relative
        else:
            print(f"No close match found for '{relative_path}'")
            return relative_path

def main():
    parser = argparse.ArgumentParser(
        description="Generate temp visual images for all scenes from a list of label filenames."
    )
    parser.add_argument(
        "--list_file",
        type=str,
        required=True,
        help="Path to the file that contains the list of label filenames."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/goose_test_1.yaml",
        help="Path to the Goose config file (default: config/goose_test_1.yaml)."
    )
    args = parser.parse_args()

    # Load the config to extract the dataset root directory.
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    root_dir = config['dataset']['root_dir']

    file_list = process_file_list(args.list_file)
    
    for filename in file_list:
        try:
            sample_image, frame_image = construct_paths(filename)
        except ValueError as e:
            print(e)
            continue
        
        # Extract the expected frame number from the original filename.
        expected_frame = extract_frame_number(filename)
        if not expected_frame:
            print(f"Could not extract frame number from {filename}")
            continue
        
        # Check for the existence of files; if missing, try to find the closest match that includes the expected frame number.
        sample_image = find_best_match_with_frame(root_dir, sample_image, expected_frame)
        frame_image = find_best_match_with_frame(root_dir, frame_image, expected_frame)
        
        # Optionally, verify that both candidate filenames contain the same frame number.
        frame_in_sample = extract_frame_number(os.path.basename(sample_image))
        frame_in_frame = extract_frame_number(os.path.basename(frame_image))
        if frame_in_sample != frame_in_frame:
            print(f"Warning: Frame numbers differ between annotation ({frame_in_sample}) and frame image ({frame_in_frame}).")
        
        # Construct the command to call the main relabeling script.
        command = (
            f"python src/main.py --config {args.config} --temp_visual "
            f"--sample_image {sample_image} --frame_image {frame_image}"
        )
        print(f"Executing: {command}")
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
