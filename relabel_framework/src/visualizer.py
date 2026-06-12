import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def convert_rgb_to_index(label_arr, palette):
    height, width, _ = label_arr.shape
    index_map = np.zeros((height, width), dtype=np.uint8)
    for idx, color in enumerate(palette):
        mask = np.all(label_arr == np.array(color, dtype=np.uint8), axis=-1)
        index_map[mask] = idx
    return index_map

def colorize_label(label_arr, palette):
    height, width = label_arr.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label_val, color in enumerate(palette):
        mask = (label_arr == label_val)
        color_image[mask] = color
    return color_image

def raw_to_seq(seg, mapping):
    out = np.zeros_like(seg, dtype=np.uint8)
    for raw_val, seq_val in mapping.items():
        out[seg == raw_val] = seq_val
    return out

def load_label_as_index(label_path, palette, id_mapping=None, id_seq=None):
    img = Image.open(label_path)
    if img.mode == "L":
        label_arr = np.array(img)
        if id_seq is not None:
            label_arr = raw_to_seq(label_arr, id_seq)
        elif id_mapping is not None:
            label_arr = raw_to_seq(label_arr, id_mapping)
        return label_arr
    else:
        rgb = np.array(img.convert("RGB"))
        return convert_rgb_to_index(rgb, palette)

def generate_comparison_image_with_two_palettes(original_label_path, new_label_path, base_palette, super_palette, id_seq=None):
    orig_label_arr = load_label_as_index(original_label_path, base_palette, id_seq=id_seq)
    orig_color = colorize_label(orig_label_arr, base_palette)
    
    new_label_arr = load_label_as_index(new_label_path, super_palette)
    new_color = colorize_label(new_label_arr, super_palette)
    
    orig_img_pil = Image.fromarray(orig_color)
    new_img_pil = Image.fromarray(new_color)
    
    width, height = orig_img_pil.size
    composite = Image.new('RGB', (width * 2, height))
    composite.paste(orig_img_pil, (0, 0))
    composite.paste(new_img_pil, (width, 0))
    
    return composite

def generate_comparison_image_with_frame_and_two_palettes(original_label_path, new_label_path, frame_path, base_palette, super_palette, id_seq=None):
    """
    Creates a composite with three panels:
      - Left: the original raw frame.
      - Middle: the original annotation (converted using id_seq if provided).
      - Right: the new relabeled annotation.
    """
    raw_img = Image.open(frame_path).convert("RGB")
    
    if id_seq is not None:
        orig_label_arr = load_label_as_index(original_label_path, base_palette, id_seq=id_seq)
    else:
        orig_label_arr = load_label_as_index(original_label_path, base_palette)
    orig_color = colorize_label(orig_label_arr, base_palette)
    orig_img_pil = Image.fromarray(orig_color)
    
    new_label_arr = load_label_as_index(new_label_path, super_palette)
    new_color = colorize_label(new_label_arr, super_palette)
    new_img_pil = Image.fromarray(new_color)
    
    # Resize annotation images to match the raw frame dimensions.
    width, height = raw_img.size
    orig_img_pil = orig_img_pil.resize((width, height))
    new_img_pil = new_img_pil.resize((width, height))
    
    composite = Image.new('RGB', (width * 3, height))
    composite.paste(raw_img, (0, 0))
    composite.paste(orig_img_pil, (width, 0))
    composite.paste(new_img_pil, (width * 2, 0))
    
    return composite

def generate_mapping_diagram_two_palettes(base_classes, groups, superclasses_mapping, base_palette, super_palette, dpi=100, present_indices=None):
    if present_indices is not None:
        indices = sorted(list(present_indices))
        base_classes = [base_classes[i] for i in indices]
        groups = [groups[i] for i in indices]
        base_palette = [base_palette[i] for i in indices]
    
    num_classes = len(base_classes)
    fig_height = max(2, num_classes * 0.3)
    fig, ax = plt.subplots(figsize=(8, fig_height), dpi=dpi)
    ax.axis('off')
    
    x_left, x_right = 0.1, 0.7
    y_positions = np.linspace(0.9, 0.1, num_classes)
    
    for i, base_class in enumerate(base_classes):
        group = groups[i]
        superclass_name = superclasses_mapping.get(str(group), "Unknown")
        y = y_positions[i]
        
        bc_color = np.array(base_palette[i]) / 255.0
        sc_color = np.array(super_palette[group]) / 255.0
        
        ax.text(x_left, y, base_class, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bc_color, edgecolor='black'))
        ax.text(x_right, y, superclass_name, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=sc_color, edgecolor='black'))
        ax.annotate("",
                    xy=(x_right - 0.05, y),
                    xytext=(x_left + 0.05, y),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.tostring_argb()
    width, height = canvas.get_width_height()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
    rgb_arr = arr[:, :, 1:4]
    plt.close(fig)
    
    return Image.fromarray(rgb_arr)

def generate_composite_visualization(original_label_path, new_label_path, base_classes, groups, superclasses_mapping, base_palette, super_palette, output_path, frame_path=None, present_indices=None, id_seq=None):
    if frame_path is not None:
        comp_img = generate_comparison_image_with_frame_and_two_palettes(original_label_path, new_label_path, frame_path, base_palette, super_palette, id_seq=id_seq)
    else:
        comp_img = generate_comparison_image_with_two_palettes(original_label_path, new_label_path, base_palette, super_palette, id_seq=id_seq)
    
    mapping_img = generate_mapping_diagram_two_palettes(base_classes, groups, superclasses_mapping, base_palette, super_palette, present_indices=present_indices)
    
    comp_width, comp_height = comp_img.size
    # Preserve aspect ratio of mapping diagram.
    mapping_width, mapping_height = mapping_img.size
    new_mapping_height = int(mapping_height * (comp_width / mapping_width))
    mapping_img = mapping_img.resize((comp_width, new_mapping_height))
    
    total_height = comp_height + new_mapping_height
    final_img = Image.new('RGB', (comp_width, total_height), (255, 255, 255))
    final_img.paste(comp_img, (0, 0))
    final_img.paste(mapping_img, (0, comp_height))
    
    final_img.save(output_path)
    print(f"Composite visualization saved to {output_path}")
