#!/usr/bin/env python
"""
YOLOv8 Cat Dataset Training Script (Optimized for GPU)
Train with GPU - Optimized for high GPU utilization
"""

import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner


def main():
    # Config file path (optimized for better GPU utilization)
    config_path = 'configs/yolov8/yolov8_s_fast_1xb16-40e_cat.py'
    config_path = 'configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_visdrone_car.py'
    

    # Load config
    cfg = Config.fromfile(config_path)

    # Set work directory
    cfg.work_dir = './work_dirs/car10'
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Load pretrained weights (COCO pretrained)
    cfg.load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'

    # ========== GPU CONFIGURATION ==========
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        # Set device to GPU (use first GPU by default)
        cfg.device = 'cuda:0'

        # Enable cuDNN benchmark for faster training
        if 'env_cfg' not in cfg:
            cfg.env_cfg = {}
        cfg.env_cfg['cudnn_benchmark'] = True

        print("=" * 60)
        print("GPU Status:")
        print(f"  CUDA Available: YES")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 60)
    else:
        print("WARNING: CUDA is NOT available!")
        print("Training will run on CPU (much slower)")
        cfg.device = 'cpu'

    # Print training config
    print("\nTraining Configuration (Optimized):")
    print(f"  Config file: {config_path}")
    print(f"  Work dir: {cfg.work_dir}")
    print(f"  Dataset: cat (1 class)")
    print(f"  Max epochs: {cfg.max_epochs}")
    print(f"  Batch size per GPU: {cfg.train_dataloader.batch_size}")
    print(f"  Number of workers: {cfg.train_dataloader.num_workers}")
    print(f"  Persistent workers: {cfg.train_dataloader.get('persistent_workers', False)}")
    print(f"  Pin memory: {cfg.train_dataloader.get('pin_memory', False)}")
    print(f"  Prefetch factor: {cfg.train_dataloader.get('prefetch_factor', 2)}")
    print(f"  Device: {cfg.device}")
    print(f"  cuDNN Benchmark: {cfg.get('env_cfg', {}).get('cudnn_benchmark', False)}")
    print("=" * 60)

    # Create Runner and start training
    print("\nInitializing Runner...")
    runner = Runner.from_cfg(cfg)

    # Verify GPU usage before training
    if cuda_available:
        model_device = next(runner.model.parameters()).device
        print(f"Model device: {model_device}")
        if model_device.type == 'cuda':
            print("Confirmed: Model is running on GPU!")
        else:
            print(f"WARNING: Model is on {model_device}, expected 'cuda'")

    print("\nStarting training...\n")
    print("TIP: Monitor GPU utilization with 'nvidia-smi -l 1' in another terminal")
    print("TIP: GPU utilization should be 70-95% during training")
    print()

    runner.train()

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
