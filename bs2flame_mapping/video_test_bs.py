import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def render_blendshape_barchart_video(bs_dir, output_video="blendshape_bars.mp4", fps=30):
    # 1. 准备文件
    valid_files = sorted([f for f in os.listdir(bs_dir) if f.endswith('.json')])
    if not valid_files:
        print(f"错误：在 {bs_dir} 中未找到任何 JSON 文件！")
        return
        
    print(f"找到 {len(valid_files)} 帧数据，准备生成动态条形图视频...")

    # 读取第一帧获取 keys
    with open(os.path.join(bs_dir, valid_files[0]), 'r') as f:
        first_frame = json.load(f)
    
    # 剔除无关项，获取纯正的 52 个表情键名
    bs_keys = sorted([k for k in first_frame.keys() if k != "_neutral"])
    
    # 2. 设置画布 (竖屏长图，方便放下 52 个条目)
    fig, ax = plt.subplots(figsize=(10, 14))
    canvas = FigureCanvas(fig)
    
    # 3. 初始化视频写入器
    # matplotlib 默认 DPI 是 100，所以 10x14 英寸就是 1000x1400 像素
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (1000, 1400))

    # 4. 逐帧绘制
    for i, fname in enumerate(valid_files):
        if i % 50 == 0:
            print(f"   -> 正在绘制第 {i}/{len(valid_files)} 帧...")

        with open(os.path.join(bs_dir, fname), 'r') as f: 
            bs_in = json.load(f)
            
        # 提取 52 个系数值 (通常范围在 0.0 到 1.0 之间)
        values = [bs_in.get(k, 0.0) for k in bs_keys]
        
        # 清空上一帧的画面
        ax.clear()
        
        # 绘制水平条形图
        y_positions = np.arange(len(bs_keys))
        ax.barh(y_positions, values, color='skyblue', edgecolor='black')
        
        # 设置坐标轴样式
        ax.set_yticks(y_positions)
        ax.set_yticklabels(bs_keys, fontsize=8)
        ax.set_xlim(0, 1.0) # Blendshape 的值通常最大为 1.0
        ax.invert_yaxis()  # 让字母 A 开头的在最上面
        ax.set_title(f"ARKit 52 Blendshapes - Frame {i:04d}", fontsize=16)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 将 matplotlib 画布转换为 OpenCV 可用的 BGR 图像
        canvas.draw()
        img_rgba = np.array(canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        # 写入视频
        video_writer.write(img_bgr)

    video_writer.release()
    plt.close(fig)
    print(f"\n大功告成！条形图视频已保存至: {os.path.abspath(output_video)}")

if __name__ == "__main__":
    # ---------------------------------------------
    # 填入你存放 json 序列的真实文件夹路径
    TARGET_BS_DIR = r"/home/abc/yg/LHM_Track/bs2flame_mapping/map_training_data/1/blendshape_params"
    OUTPUT_NAME = "blendshape_data_vis_train_data.mp4"
    # ---------------------------------------------
    
    render_blendshape_barchart_video(TARGET_BS_DIR, output_video=OUTPUT_NAME, fps=30)