#!/usr/bin/env python
import argparse
import numpy as np
from PIL import Image
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def colorize_image(label_arr, palette):
    """
    Converts a 2D label array (with integer values) into a color image using the provided palette.
    Palette is a list of [R, G, B] colors. Pixels with value i are replaced with palette[i].
    """
    height, width = label_arr.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        mask = (label_arr == idx)
        color_img[mask] = color
    return color_img

def main():
    parser = argparse.ArgumentParser(
        description="View a relabelized image with superclasses color mapping"
    )
    parser.add_argument("--input", required=True,
                        help="Path to the relabelized (grayscale) image")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML file (to obtain superclasses palette). Optional.")
    parser.add_argument("--output", default=None,
                        help="Optional output path to save the colorized image.")
    args = parser.parse_args()

    # Load the relabelized image (assumed to be grayscale)
    img = Image.open(args.input).convert("L")
    label_arr = np.array(img)

    # Get superclasses palette:
    if args.config:
        config = load_config(args.config)
        super_palette = config['superclasses']['palette']
    else:
        # Default palette for superclasses (example)
        super_palette = [
            [128, 128, 128], # Background
            [44, 160, 44], # Stable
            [255, 255, 0], # Granular
            [255, 127, 14], # Poor_Foothold
            [214, 39, 40], # High_Resistance
            [31, 119, 180], # Obstacle
        ]

    # Colorize the image using the superclasses palette.
    color_img = colorize_image(label_arr, super_palette)
    colorized = Image.fromarray(color_img)
    colorized.show()  # Display the image

    if args.output:
        colorized.save(args.output)
        print(f"Saved colorized image to {args.output}")

if __name__ == "__main__":
    main()
