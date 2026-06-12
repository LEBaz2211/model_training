import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from config_loader import load_config
from data_loader import load_list_file, ensure_dir
from relabeler import relabel_image, save_label_image
from list_generator import update_list_file
from visualizer import (
    generate_composite_visualization,
    generate_comparison_image_with_two_palettes,
    generate_mapping_diagram_two_palettes,
    load_label_as_index
)

def get_mapping_for_image(annotation_rel_path, default_mapping, config):
    """
    Determine the mapping to use based on the subfolder name.
    Optionally uses "subfolder_mappings" in the config if provided.
    """
    subfolder = os.path.basename(os.path.dirname(annotation_rel_path)).lower().strip()
    sub_mappings = config.get("classes", {}).get("subfolder_mappings", {})
    sub_mappings_lower = {k.lower().strip(): v for k, v in sub_mappings.items()}
    if subfolder in sub_mappings_lower:
        return sub_mappings_lower[subfolder]
    else:
        return default_mapping

def process_pair(root_dir, annotation_rel_path, mapping, base_palette, id_mapping, annotations_base, new_annotation_suffix):
    """
    Worker function that:
      - Opens the annotation image at root_dir/annotation_rel_path.
      - Relabels it using the provided mapping.
      - Saves the new image to a subdirectory under annotations_base.
    Returns a tuple (original_annotation_rel, new_annotation_path).
    """
    annotation_path = os.path.join(root_dir, annotation_rel_path)
    new_label_arr = relabel_image(annotation_path, mapping, base_palette=base_palette, id_mapping=id_mapping)
    base_name, ext = os.path.splitext(os.path.basename(annotation_rel_path))
    new_file_name = base_name + new_annotation_suffix + ext
    sub_dir = os.path.dirname(annotation_rel_path)
    # Use only the final part of the subdirectory to keep folder structure simple.
    new_annotation_subdir = os.path.join(annotations_base, os.path.basename(sub_dir))
    ensure_dir(new_annotation_subdir)
    new_annotation_path = os.path.join(new_annotation_subdir, new_file_name)
    save_label_image(new_label_arr, new_annotation_path)
    return (annotation_rel_path, new_annotation_path)

def main():
    parser = argparse.ArgumentParser(description="Dataset Relabeling Tool with Parallel Processing")
    parser.add_argument('--config', type=str, default='config/goose_test_1.yaml')
    parser.add_argument('--temp_visual', action='store_true', help='Generate only temporary sample visualization')
    parser.add_argument('--sample_image', type=str, default='', help='Relative path to a sample annotation image')
    parser.add_argument('--frame_image', type=str, default='', help='Relative path to a sample raw frame image')
    parser.add_argument('--sample_visuals', type=int, default=5, help='Number of sample comparison images to generate (for full pipeline)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for processing images')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_config = config['dataset']
    classes_config = config['classes']
    superclasses_config = config['superclasses']
    output_config = config['output']

    # Define output directories.
    new_annotations_dir = output_config['output_dirs']['annotations']
    new_lists_dir = output_config['output_dirs']['lists']
    visuals_dir = output_config['output_dirs']['visuals']
    for d in [new_annotations_dir, new_lists_dir, visuals_dir]:
        ensure_dir(d)
    # Store annotations base path to pass to the worker.
    annotations_base = new_annotations_dir

    base_palette = classes_config['palette']
    super_palette = superclasses_config['palette']
    default_mapping = classes_config['groups']
    superclasses_mapping = superclasses_config['mapping']
    id_mapping = classes_config.get("id_mapping", None)
    id_seq = classes_config.get("id_seq", None)
    new_annotation_suffix = output_config['new_annotation_suffix']

    root_dir = dataset_config['root_dir']

    # Temporary visualization mode remains unchanged.
    if args.temp_visual:
        if args.sample_image:
            annotation_rel_path = args.sample_image
            orig_annotation_path = os.path.join(root_dir, annotation_rel_path)
        else:
            train_list_path = os.path.join(root_dir, dataset_config['list_files']['train'])
            pairs = load_list_file(train_list_path)
            if not pairs:
                print("No entries in train.txt!")
                return
            _, annotation_rel_path = pairs[0]
            orig_annotation_path = os.path.join(root_dir, annotation_rel_path)
        mapping = get_mapping_for_image(annotation_rel_path, default_mapping, config)
        new_label_arr = relabel_image(orig_annotation_path, mapping, base_palette=base_palette, id_mapping=id_mapping)
        temp_new_path = "temp_new_annotation.png"
        save_label_image(new_label_arr, temp_new_path)
        present_indices = set(np.unique(load_label_as_index(orig_annotation_path, base_palette, id_seq=id_seq)))
        ensure_dir(visuals_dir)
        base_name = os.path.splitext(os.path.basename(orig_annotation_path))[0]
        composite_path = os.path.join(visuals_dir, f"{base_name}_temp_composite_visual.png")
        frame_path = os.path.join(root_dir, args.frame_image) if args.frame_image else None
        generate_composite_visualization(
            original_label_path=orig_annotation_path,
            new_label_path=temp_new_path,
            base_classes=classes_config['base_classes'],
            groups=mapping,
            superclasses_mapping=superclasses_mapping,
            base_palette=base_palette,
            super_palette=super_palette,
            output_path=composite_path,
            frame_path=frame_path,
            present_indices=present_indices,
            id_seq=id_seq
        )
        os.remove(temp_new_path)
        return

    # FULL PIPELINE: First, load all pairs from the list files.
    list_files = dataset_config['list_files']
    all_tasks = []  # Each task is a tuple: (annotation_rel_path, mapping)
    for list_name, list_file in list_files.items():
        list_file_path = os.path.join(root_dir, list_file)
        pairs = load_list_file(list_file_path)
        print(f"Processing {len(pairs)} entries from {list_name}...")
        for frame_rel_path, annotation_rel_path in pairs:
            mapping = get_mapping_for_image(annotation_rel_path, default_mapping, config)
            all_tasks.append((annotation_rel_path, mapping))

    # Process the relabeling in parallel.
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {
            executor.submit(
                process_pair,
                root_dir,
                annotation_rel_path,
                mapping,
                base_palette,
                id_mapping,
                annotations_base,
                new_annotation_suffix
            ): annotation_rel_path for annotation_rel_path, mapping in all_tasks
        }
        for future in as_completed(future_to_task):
            annotation_rel_path = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Error processing {annotation_rel_path}: {exc}")

    # Update list files sequentially (this is typically fast).
    for list_name, list_file in list_files.items():
        list_file_path = os.path.join(root_dir, list_file)
        new_list_file_name = output_config['new_list_prefix'] + os.path.basename(list_file)
        new_list_file_path = os.path.join(new_lists_dir, new_list_file_name)
        update_list_file(list_file_path, new_annotation_suffix, new_list_file_path)
    
    # Generate sample visual comparisons from the train list.
    sample_count = 0
    train_list_path = os.path.join(root_dir, list_files['train'])
    pairs = load_list_file(train_list_path)
    for frame_rel_path, annotation_rel_path in pairs:
        if sample_count >= args.sample_visuals:
            break
        orig_annotation_path = os.path.join(root_dir, annotation_rel_path)
        base_name, ext = os.path.splitext(os.path.basename(annotation_rel_path))
        new_file_name = base_name + new_annotation_suffix + ext
        sub_dir = os.path.dirname(annotation_rel_path)
        new_annotation_subdir = os.path.join(new_annotations_dir, os.path.basename(sub_dir))
        new_annotation_path = os.path.join(new_annotation_subdir, new_file_name)
        comp_output_path = os.path.join(visuals_dir, f"comparison_{sample_count+1}.png")
        comp_img = generate_comparison_image_with_two_palettes(orig_annotation_path, new_annotation_path, base_palette, super_palette, id_seq=id_seq)
        comp_img.save(comp_output_path)
        sample_count += 1

    mapping_summary_path = os.path.join(visuals_dir, "mapping_summary.png")
    mapping_img = generate_mapping_diagram_two_palettes(classes_config['base_classes'], default_mapping, superclasses_mapping, base_palette, super_palette)
    mapping_img.save(mapping_summary_path)
    
    print("Relabeling complete. New annotations, list files, and visuals are available in the output directories.")

if __name__ == "__main__":
    main()
