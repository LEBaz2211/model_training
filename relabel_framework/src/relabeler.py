import numpy as np
from PIL import Image

def convert_rgb_to_index(label_arr, palette):
    """
    Converts an RGB label image (H x W x 3) to a 2D index image by matching exact colors.
    If a pixel doesn't match any palette entry, it remains 0.
    """
    height, width, _ = label_arr.shape
    index_map = np.zeros((height, width), dtype=np.uint8)
    for idx, color in enumerate(palette):
        mask = np.all(label_arr == np.array(color, dtype=np.uint8), axis=-1)
        index_map[mask] = idx
    return index_map

def raw_to_seq(seg, mapping):
    """
    Converts a raw label image (2D array) using the provided mapping dictionary.
    For each pixel value raw_val in seg, it assigns mapping[raw_val].
    """
    out = np.zeros_like(seg, dtype=np.uint8)
    for raw_val, seq_val in mapping.items():
        out[seg == raw_val] = seq_val
    return out

def relabel_image(annotation_path, groups, base_palette=None, id_mapping=None):
    """
    Reads an annotation image and creates a new label array by mapping each base class index 
    to its corresponding group.
    
    Behavior:
      - If the image mode is "P" (paletted), it is converted to "L" to get grayscale values.
      - If the image mode is not "L" (or "P"), and a base_palette is provided, it is converted to RGB 
        and then to indices via the palette.
      - If the image mode is "L" and id_mapping is provided, the raw values are converted using id_mapping.
      - Otherwise, the grayscale image is assumed to already contain sequential indices, which are then remapped.
    
    Parameters:
      annotation_path: Path to the annotation image.
      groups: A list where groups[i] is the new group for base class i.
      base_palette: List of RGB colors for each base class (required for non-grayscale images).
      id_mapping: (Optional) Dictionary mapping raw label values to new group values.
                  Use this for datasets like Goose.
    """
    img = Image.open(annotation_path)
    
    # If image is paletted, convert to L to get numeric indices.
    if img.mode == "P":
        img = img.convert("L")
        label_arr = np.array(img)
        if id_mapping is not None:
            # For Goose: convert raw values using id_mapping.
            return raw_to_seq(label_arr, id_mapping)
    # For non-grayscale images (e.g., RGB), use the base_palette.
    elif img.mode != "L":
        if base_palette is None:
            raise ValueError("Non-grayscale image detected. Provide base_palette for conversion.")
        img = img.convert("RGB")
        rgb = np.array(img)
        label_arr = convert_rgb_to_index(rgb, base_palette)
    else:
        # Image mode is "L" but no id_mapping provided.
        label_arr = np.array(img)
        if id_mapping is not None:
            return raw_to_seq(label_arr, id_mapping)
    
    # Create new label array by mapping sequential indices.
    new_label_arr = np.zeros_like(label_arr, dtype=np.uint8)
    for old_class, new_class in enumerate(groups):
        new_label_arr[label_arr == old_class] = new_class
    return new_label_arr

def save_label_image(label_arr, save_path):
    """
    Saves a numpy array as a grayscale image.
    """
    img = Image.fromarray(label_arr.astype('uint8'))
    img.save(save_path)
