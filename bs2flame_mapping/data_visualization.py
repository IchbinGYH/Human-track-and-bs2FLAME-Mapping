import os
import json
import glob
import matplotlib.pyplot as plt
import math

# 🌟 标准 52 维列表 (标准的 ARKit 52)
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

def plot_blendshape_distributions(json_dir, output_image_path):
    print(f"📂 正在扫描目录: {json_dir}")
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    if not json_files:
        print("❌ 未找到任何 .json 文件，请检查路径！")
        return

    print(f"🔍 共找到 {len(json_files)} 帧数据，正在加载并提取特征...")

    # 1. 初始化数据容器，为每个 Blendshape 创建一个空列表
    data_dict = {name: [] for name in STANDARD_NAMES}

    # 2. 逐文件读取数据
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                bs_data = json.load(f)
                # 提取 52 个维度的值，如果没有找到则默认给 0.0
                for name in STANDARD_NAMES:
                    val = bs_data.get(name, 0.0)
                    data_dict[name].append(val)
            except Exception as e:
                print(f"读取 {json_path} 时出错: {e}")

    print("✅ 数据提取完成，正在生成 52 宫格分布图...")

    # 3. 准备绘图 (8 行 7 列 = 56 个子图，画 52 个，空 4 个)
    num_cols = 7
    num_rows = math.ceil(len(STANDARD_NAMES) / num_cols)
    
    # 设置画布大小 (宽 24 英寸, 高 20 英寸)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 20))
    axes = axes.flatten() # 展平方便遍历

    # 4. 开始绘制每个特征的直方图
    for i, name in enumerate(STANDARD_NAMES):
        ax = axes[i]
        values = data_dict[name]
        
        # 绘制直方图，分为 40 个区间 (bins)
        # log=True 可以把 Y 轴变成对数坐标，方便看清极端值；如果不需要对数坐标，把 log 删掉即可
        ax.hist(values, bins=40, range=(0.0, 1.0), color='royalblue', edgecolor='black', alpha=0.7, log=False)
        
        # 设置标题和样式
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlim(0.0, 1.0) # 固定 X 轴范围为 0 到 1
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. 隐藏多余的空白子图 (第 53-56 个)
    for j in range(len(STANDARD_NAMES), len(axes)):
        fig.delaxes(axes[j])

    # 6. 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"🎉 绘制成功！分布图已保存至: {os.path.abspath(output_image_path)}")

# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 填写你的 json 文件夹路径
    INPUT_JSON_DIR = "/home/abc/yg/LHM_Track/bs2flame_mapping/map_test_data/3/blendshape_param"
    
    # 输出的高清全景图路径
    OUTPUT_IMAGE = "/home/abc/yg/LHM_Track/bs2flame_mapping/data_visualization/blendshape_distribution_52_a2e.png"
    
    plot_blendshape_distributions(INPUT_JSON_DIR, OUTPUT_IMAGE)