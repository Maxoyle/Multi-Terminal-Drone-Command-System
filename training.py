from ultralytics import YOLO

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'

import torch
torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True  # 固定输入尺寸时加速卷积

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 减少显存碎片

def main():
    # Load a model
    model = YOLO("yolo11m.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(
        data="data.yaml",
        cache=True,
        epochs = 5000,
        imgsz = 640,
        batch = 32,  # 实际batch
        #rect=True,
        #fraction = 0.1,
        #scale = 0.5,
        amp = True,
        #mosaic=0.5,  # 调整数据增强参数
        #mixup=0.5,
        #workers = 12, # 建议设为 CPU 核心数的 70-80%
        #copy_paste=0.1,            # 添加更适合小目标的增强
        optimizer='Adam',         # 尝试不同优化器
        #overlap_mask=True,         # 对小目标重要
        #lr0=0.001,                 # 明确设置学习率
        #hsv_h=0.01,               # 降低颜色扰动
        #hsv_s=0.5,
        #hsv_v=0.4,
        #degrees=5.0,              # 减少几何变换
        #translate=0.05,
        #shear=2.0,
        #fliplr=0.3,               # 降低水平翻转概率
        #patience=100,
        val = True
    )

    print("Training completed. Results:" ,results)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # 仅在 Windows 下需要
    main()