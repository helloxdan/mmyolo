# Copyright (c) OpenMMLab. All rights reserved.
"""Extract subset from VisDrone dataset by specified classes.

Filters YOLO txt labels and images, keeping only specified class annotations.
Outputs results to <data_root>/sub by default.

Usage:
    python tools/misc/extract_sub_visdrone.py --data-root data/VisDrone_car --classes car
    python tools/misc/extract_sub_visdrone.py --data-root data/VisDrone_car --classes car,truck --out-dir data/VisDrone_car/sub_car_truck
"""

import argparse
import os
import os.path as osp
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Extract subset from VisDrone')
    parser.add_argument('--data-root', required=True, help='root path of VisDrone dataset')
    parser.add_argument('--classes', type=str, required=True, help='classes to keep, comma-separated (e.g. car,truck)')
    parser.add_argument('--out-dir', default='/sub', help='output sub-directory relative to data-root (default: /sub)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root = args.data_root
    out_dir = osp.join(root, args.out_dir.lstrip('/'))

    # Parse comma-separated classes
    classes = [c.strip() for c in args.classes.split(',')]
    assert out_dir != root, \
        'The output directory must be different from the source directory!'

    # Load class mapping
    classes_file = osp.join(root, 'classes.txt')
    with open(classes_file, 'r') as f:
        all_classes = [line.strip() for line in f if line.strip()]

    # Get target class ids
    target_ids = set()
    for cls_name in classes:
        if cls_name in all_classes:
            target_ids.add(all_classes.index(cls_name))
        else:
            print(f'Warning: class "{cls_name}" not found in classes.txt, skipping')

    if not target_ids:
        print('Error: no valid classes found')
        return

    print(f'Keeping class ids {target_ids} for classes: {classes}')

    # Create output directories
    out_images = osp.join(out_dir, 'images')
    out_labels = osp.join(out_dir, 'labels')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # Process labels and images
    labels_dir = osp.join(root, 'labels')
    images_dir = osp.join(root, 'images')

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith('.txt'))
    total = len(label_files)
    kept_count = 0
    empty_count = 0

    for i, label_file in enumerate(label_files):
        label_path = osp.join(labels_dir, label_file)
        img_path = osp.join(images_dir, osp.splitext(label_file)[0] + '.jpg')

        # Read and filter annotations
        new_lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in target_ids:
                    # Remap class id to 0
                    parts[0] = str(sorted(target_ids).index(cls_id))
                    new_lines.append(' '.join(parts))

        if not new_lines:
            empty_count += 1
            sys.stdout.write(f'\rProcessing: {i + 1}/{total}')
            sys.stdout.flush()
            continue

        # Copy image
        shutil.copy(img_path, osp.join(out_images, osp.splitext(label_file)[0] + '.jpg'))

        # Write filtered labels
        with open(osp.join(out_labels, label_file), 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

        kept_count += 1
        sys.stdout.write(f'\rProcessing: {i + 1}/{total}')
        sys.stdout.flush()

    # Write sub classes.txt
    sub_classes = [cls for cls in classes if cls in all_classes]
    with open(osp.join(out_dir, 'classes.txt'), 'w') as f:
        for cls_name in sub_classes:
            f.write(cls_name + '\n')

    print(f'\nDone! Kept {kept_count} images (skipped {empty_count} without target classes)')
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()
