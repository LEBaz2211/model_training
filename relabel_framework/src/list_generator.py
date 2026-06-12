import os

def update_list_file(original_list_path, new_suffix, output_list_path):
    """
    Reads an original list file and writes a new list file where the annotation
    file names are updated with the new suffix.
    """
    updated_lines = []
    with open(original_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            frame_path, annotation_path = parts
            base, ext = os.path.splitext(annotation_path)
            new_annotation_path = base + new_suffix + ext
            updated_line = f"{frame_path} {new_annotation_path}"
            updated_lines.append(updated_line)
    with open(output_list_path, 'w') as f:
        for line in updated_lines:
            f.write(line + "\n")
