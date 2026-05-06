# ==========================================
# 依然是最重要的第一步：强制使用 EGL 离屏渲染
# ==========================================
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import json
import torch
import numpy as np
import trimesh
import pyrender
import cv2
import smplx

# ==========================================
# 视频渲染主程序 (单路 FLAME 渲染)
# ==========================================
def render_single_video(flame_dir, flame_model_path, output_video="single_output.mp4", fps=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 正在初始化设备: {device}")

    # --- A. 加载 FLAME 模型 ---
    print("2. 正在加载 FLAME 模型...")
    flame_model = smplx.create(flame_model_path, model_type='flame', 
                               use_face_contour=False, 
                               num_expression_coeffs=100).to(device)
    faces = flame_model.faces

    # --- B. 准备需要处理的文件序列 ---
    # 获取并按名字排序 (非常重要，保证视频帧顺序正确)
    valid_files = sorted([f for f in os.listdir(flame_dir) if f.endswith('.json')])
    
    if not valid_files:
        print(f"错误：在 {flame_dir} 中未找到任何 JSON 文件！")
        return

    print(f"3. 找到 {len(valid_files)} 帧有效数据，准备渲染视频...")

    # --- C. 设置 PyRender 场景 (只初始化一次，避免内存泄漏) ---
    renderer = pyrender.OffscreenRenderer(512, 512)
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.35],
       [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # --- D. 设置 OpenCV VideoWriter ---
    # 分辨率: 宽 512, 高 512 (单画面)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (512, 512))

    # --- E. 开始逐帧渲染 ---
    with torch.no_grad():
        for i, fname in enumerate(valid_files):
            # 进度打印
            if i % 50 == 0:
                print(f"   -> 正在处理第 {i}/{len(valid_files)} 帧...")

            # 1. 读取数据
            with open(os.path.join(flame_dir, fname), 'r') as f: 
                fl_data = json.load(f)
            
            # 2. 提取 FLAME 参数 Tensor
            exp = torch.tensor([fl_data['expcode']], dtype=torch.float32).to(device)
            jaw = torch.tensor([fl_data['posecode'][3:6]], dtype=torch.float32).to(device)
            eye = torch.tensor([fl_data['eyecode']], dtype=torch.float32).to(device)

            # 3. 通过 FLAME 得到 3D 顶点
            vertices = flame_model(expression=exp, jaw_pose=jaw, eye_pose=eye).vertices.detach().cpu().numpy().squeeze()

            # 4. 渲染成图像
            mesh = trimesh.Trimesh(vertices, faces)
            mesh.visual.vertex_colors = [0.8, 0.8, 0.8, 1.0]
            py_mesh = pyrender.Mesh.from_trimesh(mesh)
            
            node = scene.add(py_mesh)
            color, _ = renderer.render(scene)
            scene.remove_node(node) # 拍完照必须把旧的人脸移出场景！

            # 5. OpenCV 转换与写入
            img_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            
            # 贴上文字标签 (可选，不需要可注释掉)
            cv2.putText(img_bgr, 'FLAME Render', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            video_writer.write(img_bgr)

    # 释放资源
    video_writer.release()
    renderer.delete()
    print(f"\n大功告成！视频已保存至: {os.path.abspath(output_video)}")


if __name__ == "__main__":
    # ---------------------------------------------
    # 1. 填入具体包含 json 文件的路径 (精确到 flame_params 这一级)
    TARGET_FLAME_DIR = r"/home/abc/yg/LHM_Track/bs2flame_mapping/map_test_data/3/flame_param"
    
    # 2. FLAME 模型基底路径
    FLAME_MODEL_DIR = r"/home/abc/yg/models"
    
    # 3. 输出视频名称与帧率
    OUTPUT_NAME = "single_flame_render.mp4"
    # ---------------------------------------------
    
    render_single_video(TARGET_FLAME_DIR, FLAME_MODEL_DIR, output_video=OUTPUT_NAME, fps=30)