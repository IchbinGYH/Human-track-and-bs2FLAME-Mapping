import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import math

# 🌟 标准 52 维列表
STANDARD_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut"
]

def plot_adaptive_blendshape_distributions(json_dir, output_image_path):
    print(f"📂 正在扫描目录: {json_dir}")
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    if not json_files:
        print("❌ 未找到任何 .json 文件，请检查路径！")
        return

    print(f"🔍 共加载 {len(json_files)} 帧数据，正在进行自适应统计...")

    data_dict = {name: [] for name in STANDARD_NAMES}

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                bs_data = json.load(f)
                for name in STANDARD_NAMES:
                    data_dict[name].append(bs_data.get(name, 0.0))
            except Exception as e:
                pass

    print("✅ 数据提取完成，正在生成极客版 52 宫格分布图...")

    num_cols = 7
    num_rows = math.ceil(len(STANDARD_NAMES) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(26, 22))
    axes = axes.flatten()

    for i, name in enumerate(STANDARD_NAMES):
        ax = axes[i]
        # 转换为 NumPy 数组方便统计
        values = np.array(data_dict[name])
        
        if len(values) == 0:
            continue

        # 核心统计量
        zero_ratio = np.mean(values == 0.0) * 100  # 计算纯 0 值的百分比
        val_max = np.max(values)
        
        # 💡 自适应计算 X 轴范围
        # 如果全为0，给一个最小的显示范围；否则按照最大值留出 10% 的余量
        x_max_plot = val_max * 1.1 if val_max > 0 else 0.1
        # 但如果不小心超过了 1.0，最高强制截断到 1.05
        if x_max_plot > 1.0: x_max_plot = 1.05

        # 💡 核心绘图：自适应 bins + 对数 Y 轴
        # range 限定在实际数据的分布范围内，避免计算无意义的空白区
        ax.hist(values, bins=35, range=(0.0, max(val_max, 0.01)), 
                color='mediumseagreen', edgecolor='black', alpha=0.75, log=True)
        
        # 设置坐标轴范围
        ax.set_xlim(-0.01, x_max_plot)
        
        # 标题
        ax.set_title(name, fontsize=10, fontweight='bold', color='darkblue')

        # 💡 数据面板：在右上角悬浮显示 Zeros% 和 Max 值
        info_text = f"Zeros: {zero_ratio:.1f}%\nMax: {val_max:.3f}"
        ax.text(0.95, 0.95, info_text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=9, color='firebrick', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='ivory', ec='gray', alpha=0.9))

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 隐藏空白的子图
    for j in range(len(STANDARD_NAMES), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"🎉 大功告成！完美分布图已保存至: {os.path.abspath(output_image_path)}")

if __name__ == "__main__":
    # 填写你的 json 文件夹路径
    INPUT_JSON_DIR = "/home/abc/yg/LHM_Track/bs2flame_mapping/map_training_data/1/blendshape_params"
    
    # 输出的高清全景图路径
    OUTPUT_IMAGE = "/home/abc/yg/LHM_Track/bs2flame_mapping/data_visualization/blendshape_distribution_52_train1.png"
    
    plot_adaptive_blendshape_distributions(INPUT_JSON_DIR, OUTPUT_IMAGE)