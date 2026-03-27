# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMYOLO is OpenMMLab's YOLO series toolbox for object detection, built on PyTorch, MMDetection, MMEngine, and MMCV. It implements multiple YOLO variants (YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOX, RTMDet, PPYOLOE) with a unified modular design.

## Installation

```bash
# Install dependencies
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"
pip install -r requirements/albu.txt  # For albumentations transforms
mim install -v -e .  # Install MMYOLO in editable mode
```

## Common Commands

### Training
```bash
# Single GPU training
python tools/train.py configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py

# Multi-GPU training (recommended for YOLO)
torchrun --nproc_per_node=8 tools/train.py configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py --launcher pytorch

# Resume training
python tools/train.py <config> --resume <checkpoint_path>
python tools/train.py <config> --resume auto  # Auto-resume from latest checkpoint

# AMP training
python tools/train.py <config> --amp
```

### Testing/Inference
```bash
# Test with evaluation
python tools/test.py <config> <checkpoint>

# Test with visualization
python tools/test.py <config> <checkpoint> --show-dir ./results

# Test time augmentation
python tools/test.py <config> <checkpoint> --tta

# Deploy mode (convert reparameterized models)
python tools/test.py <config> <checkpoint> --deploy
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models/test_dense_heads/test_yolov8_head.py

# Run with coverage
pytest --cov=mmyolo --cov-report=html
```

## Architecture

### Registry System
MMYOLO uses MMEngine's registry system for modular components. All registries are defined in `mmyolo/registry.py`:
- `MODELS`: Backbones, necks, heads, detectors
- `DATASETS`: Dataset implementations
- `TRANSFORMS`: Data augmentation/transforms
- `HOOKS`: Training hooks (param schedulers, EMA, etc.)
- `TASK_UTILS`: Assigners, coders, samplers

### Module Structure
- **`mmyolo/models/detectors/`**: `YOLODetector` - base detector class extending MMDetection's `SingleStageDetector`
- **`mmyolo/models/backbones/`**: YOLO-specific backbones (YOLOv5CSPDarknet, YOLOv8CSPDarknet, CSPNeXt, EfficientRep, etc.)
- **`mmyolo/models/necks/`**: Feature pyramid networks (YOLOv5PAFPN, YOLOv8PAFPN, CSPNeXtPAFPN, etc.)
- **`mmyolo/models/dense_heads/`**: Detection heads (YOLOv5Head, YOLOv8Head, RTMDetHead, etc.)
- **`mmyolo/models/task_modules/assigners/`**: Label assigners for training (BatchTaskAlignedAssigner, BatchATSSAssigner)
- **`mmyolo/models/task_modules/coders/`**: Bbox coders (DistancePointBBoxCoder, YOLOv5BBoxCoder)
- **`mmyolo/datasets/`**: Dataset wrappers (YOLOv5CocoDataset, YOLOv5VOCDataset)
- **`mmyolo/datasets/transforms/`**: Custom transforms (YOLOv5HSVRandomAug, Mosaic, YOLOv5RandomAffine)
- **`mmyolo/engine/`**: Training hooks and optimizer constructors

### Config System
Configs use hierarchical inheritance with `_base_` imports. Key config sections:
- **`model`**: Backbone, neck, head configuration with scaling factors (`deepen_factor`, `widen_factor`)
- **`train_dataloader/val_dataloader`**: Data pipeline with transforms
- **`optim_wrapper`**: Optimizer with gradient clipping and custom constructors
- **`param_scheduler`**: Learning rate scheduling (often via YOLOv5ParamSchedulerHook)
- **`default_hooks/custom_hooks`**: Checkpointing, EMA, pipeline switching

## Key Development Patterns

### Adding New YOLO Variants
1. Create backbone in `mmyolo/models/backbones/`
2. Create neck in `mmyolo/models/necks/` (typically extends `BaseYOLONeck`)
3. Create head in `mmyolo/models/dense_heads/` (typically includes head_module + assigner)
4. Register components with `@MODELS.register_module()`
5. Create config in `configs/<variant>/`

### Custom Data Augmentation
Add transforms to `mmyolo/datasets/transforms/` and register with `@TRANSFORMS.register_module()`. YOLO-specific transforms like Mosaic require special handling in the collate function.

### Label Assigners
YOLO variants use different label assignment strategies. Key assigners:
- `BatchTaskAlignedAssigner`: YOLOv8/RTMDet (TAL algorithm)
- `BatchATSSAssigner`: YOLOX/RTMDet
- `BatchYTALAssigner`: YOLOv7
- `BatchDynamicLabelAssigner`: YOLOv6

### Cross-Library Usage
MMYOLO can use backbones from MMDetection and MMPreTrain via registry prefix:
- `type='mmdet.ResNet'` for MMDetection models
- `type='mmcls.ResNet'` for MMPreTrain models

## Important Notes

- **SyncBatchNorm**: YOLO models default to `use_syncbn=True` for multi-GPU training
- **Reparameterization**: Models like YOLOv6/RTMDet require `SwitchToDeployHook` for inference
- **Scaling factors**: Most models use `deepen_factor` and `widen_factor` for network scaling
- **Data preprocessing**: Uses `YOLOv5DetDataPreprocessor` with normalized mean=[128,128,128], std=[128,128,128]
- **Anchor-free**: Most modern YOLO variants (v5+, v8, RTMDet) are anchor-free using point generators
