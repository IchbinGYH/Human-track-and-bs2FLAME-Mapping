# ==========================================
# 依然是最重要的第一步：强制使用 EGL 离屏渲染
# ==========================================
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import json
import torch
import torch.nn as nn
import numpy as np
import trimesh
import pyrender
import cv2
import smplx

# ==========================================
# 1. 网络结构定义 (保持不变)
# ==========================================
class MappingMLP(nn.Module):
    def __init__(self, input_dim=52, output_dim=100, hidden_layers=[256, 128]):
        super(MappingMLP, self).__init__()
        layers = []
        in_features = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) 
            layers.append(nn.ReLU())
            in_features = h_dim
        layers.append(nn.Linear(in_features, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x): 
        return self.network(x)

# ==========================================
# 2. 视频渲染主程序
# ==========================================
def render_comparison_video(folder_dir, weights_paths, flame_model_path, output_video="output_comparison.mp4", fps=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 正在初始化设备: {device}")

    # --- A. 加载 MLP 模型 ---
    print("2. 正在加载 MLP 权重...")
    m_exp = MappingMLP(52, 100, [256, 128]).to(device)
    m_jaw = MappingMLP(52, 3, [128, 64]).to(device)
    m_eye = MappingMLP(52, 6, [128, 64]).to(device)
    
    m_exp.load_state_dict(torch.load(weights_paths['exp'], map_location=device))
    m_jaw.load_state_dict(torch.load(weights_paths['jaw'], map_location=device))
    m_eye.load_state_dict(torch.load(weights_paths['eye'], map_location=device))
    m_exp.eval(); m_jaw.eval(); m_eye.eval()

    # --- B. 加载 FLAME 模型 ---
    print("3. 正在加载 FLAME 模型...")
    flame_model = smplx.create(flame_model_path, model_type='flame', 
                               use_face_contour=False, 
                               num_expression_coeffs=100).to(device)
    faces = flame_model.faces

    # --- C. 准备需要处理的文件序列 ---
    flame_dir = os.path.join(folder_dir, "flame_params")
    bs_dir = os.path.join(folder_dir, "blendshape_params")
    
    # 获取交集并按名字排序 (非常重要，保证视频帧顺序正确)
    valid_files = sorted(list(set(os.listdir(flame_dir)) & set(os.listdir(bs_dir))))
    valid_files = [f for f in valid_files if f.endswith('.json')]
    
    if not valid_files:
        print("错误：未找到任何配对的 JSON 文件！")
        return

    print(f"4. 找到 {len(valid_files)} 帧有效配对数据，准备渲染视频...")

    # --- D. 设置 PyRender 场景 (只初始化一次，避免内存泄漏) ---
    renderer = pyrender.OffscreenRenderer(512, 512)
    scene_gt = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    scene_pred = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.35],
       [0.0, 0.0, 0.0, 1.0]
    ])
    scene_gt.add(camera, pose=camera_pose)
    scene_pred.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene_gt.add(light, pose=camera_pose)
    scene_pred.add(light, pose=camera_pose)

    # --- E. 设置 OpenCV VideoWriter ---
    # 分辨率: 宽 1024 (512+512), 高 512
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (1024, 512))

    bs_keys = None

    # --- F. 开始逐帧渲染 ---
    with torch.no_grad():
        for i, fname in enumerate(valid_files):
            # 进度打印
            if i % 50 == 0:
                print(f"   -> 正在处理第 {i}/{len(valid_files)} 帧...")

            # 1. 读取数据
            with open(os.path.join(bs_dir, fname), 'r') as f: bs_in = json.load(f)
            with open(os.path.join(flame_dir, fname), 'r') as f: fl_gt = json.load(f)
            
            if bs_keys is None: bs_keys = sorted(bs_in.keys())
            
            # 2. GT 参数 Tensor
            gt_exp = torch.tensor([fl_gt['expcode']], dtype=torch.float32).to(device)
            gt_jaw = torch.tensor([fl_gt['posecode'][3:6]], dtype=torch.float32).to(device)
            gt_eye = torch.tensor([fl_gt['eyecode']], dtype=torch.float32).to(device)
            
            # 3. MLP 预测参数
            x_bs = torch.tensor([[bs_in[k] for k in bs_keys]], dtype=torch.float32).to(device)
            pred_exp = m_exp(x_bs)
            pred_jaw = m_jaw(x_bs)
            pred_eye = m_eye(x_bs)

            # 4. 通过 FLAME 得到 3D 顶点
            gt_vertices = flame_model(expression=gt_exp, jaw_pose=gt_jaw, eye_pose=gt_eye).vertices.detach().cpu().numpy().squeeze()
            pred_vertices = flame_model(expression=pred_exp, jaw_pose=pred_jaw, eye_pose=pred_eye).vertices.detach().cpu().numpy().squeeze()

            # 5. 渲染成图像
            # -- GT --
            mesh_gt = trimesh.Trimesh(gt_vertices, faces)
            mesh_gt.visual.vertex_colors = [0.8, 0.8, 0.8, 1.0]
            py_mesh_gt = pyrender.Mesh.from_trimesh(mesh_gt)
            node_gt = scene_gt.add(py_mesh_gt)
            color_gt, _ = renderer.render(scene_gt)
            scene_gt.remove_node(node_gt) # 拍完照必须把旧的人脸移出场景！

            # -- Pred --
            mesh_pred = trimesh.Trimesh(pred_vertices, faces)
            mesh_pred.visual.vertex_colors = [0.8, 0.8, 0.8, 1.0]
            py_mesh_pred = pyrender.Mesh.from_trimesh(mesh_pred)
            node_pred = scene_pred.add(py_mesh_pred)
            color_pred, _ = renderer.render(scene_pred)
            scene_pred.remove_node(node_pred)

            # 6. OpenCV 拼接与压制
            img_gt_bgr = cv2.cvtColor(color_gt, cv2.COLOR_RGB2BGR)
            img_pred_bgr = cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR)
            
            # 贴上文字标签
            cv2.putText(img_gt_bgr, 'Ground Truth (FLAME)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img_pred_bgr, 'MLP Predicted', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            combined_frame = np.hstack((img_gt_bgr, img_pred_bgr))
            video_writer.write(combined_frame)

    # 释放资源
    video_writer.release()
    renderer.delete()
    print(f"\n大功告成！视频已保存至: {os.path.abspath(output_video)}")


if __name__ == "__main__":
    # ---------------------------------------------
    # 1. 填入你要测试的那个特定文件夹 (比如包含一整个动作序列的 1 号文件夹)
    # 注意：是指向带有 blendshape_params 和 flame_params 的父文件夹
    TARGET_FOLDER = r"/home/abc/yg/LHM_Track/bs2flame_mapping/map_test_data/1/test"
    
    # 2. 模型与权重路径
    FLAME_MODEL_DIR = r"/home/abc/yg/models"
    WEIGHTS = {
        'exp': "mlp_exp_weights.pth", 
        'jaw': "mlp_jaw_weights.pth", 
        'eye': "mlp_eye_weights.pth"
    }
    
    # 3. 输出视频名称与帧率 (可根据你原始视频的帧率如 30 或 60 修改 fps)
    OUTPUT_NAME = "demo_comparison3.mp4"
    # ---------------------------------------------
    
    render_comparison_video(TARGET_FOLDER, WEIGHTS, FLAME_MODEL_DIR, output_video=OUTPUT_NAME, fps=30)