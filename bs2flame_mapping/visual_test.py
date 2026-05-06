# ==========================================
# 极其重要：这几行必须在所有代码的最前面！
# 强制 PyRender 在 Linux 服务器上使用 EGL 离屏渲染
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
# 1. 渲染核心函数
# ==========================================
def render_mesh_to_image(vertices, faces, color=(0.8, 0.8, 0.8)):
    """将3D Mesh渲染为2D图像"""
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.visual.vertex_colors = [0.8, 0.8, 0.8, 1.0] # 赋予灰色
    
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(render_mesh)
    
    # 设置相机位置 (正对人脸)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
       [1.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 0.0, 0.0],
       [0.0, 0.0, 1.0, 0.35], # Z轴距离，0.35米刚好能看清整张脸
       [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # 添加方向光
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # 离屏渲染 (分辨率 512x512)
    r = pyrender.OffscreenRenderer(512, 512)
    color_img, depth = r.render(scene)
    r.delete()
    
    return color_img

# ==========================================
# 2. 主测试与可视化逻辑
# ==========================================
def visualize_results(results_dir, flame_model_path, output_dir="render_output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"初始化 FLAME 模型 (路径: {flame_model_path})...")
    
    flame_model = smplx.create(flame_model_path, model_type='flame', 
                               use_face_contour=False, 
                               num_expression_coeffs=100).to(device)
    
    faces = flame_model.faces

    result_files = [f for f in os.listdir(results_dir) if f.startswith('res_') and f.endswith('.json')]
    print(f"找到 {len(result_files)} 个测试结果，开始渲染...")

    for fname in result_files:
        with open(os.path.join(results_dir, fname), 'r') as f:
            data = json.load(f)
            
        gt = data['ground_truth_flame']
        pred = data['predicted_flame']
        
        gt_exp = torch.tensor([gt['exp']], dtype=torch.float32).to(device)
        gt_jaw = torch.tensor([gt['jaw']], dtype=torch.float32).to(device)
        gt_eye = torch.tensor([gt['eye']], dtype=torch.float32).to(device)
        
        pred_exp = torch.tensor([pred['exp']], dtype=torch.float32).to(device)
        pred_jaw = torch.tensor([pred['jaw']], dtype=torch.float32).to(device)
        pred_eye = torch.tensor([pred['eye']], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            gt_vertices = flame_model(expression=gt_exp, jaw_pose=gt_jaw, eye_pose=gt_eye).vertices.detach().cpu().numpy().squeeze()
            pred_vertices = flame_model(expression=pred_exp, jaw_pose=pred_jaw, eye_pose=pred_eye).vertices.detach().cpu().numpy().squeeze()

        try:
            print(f"正在渲染照片: {fname}...")
            img_gt = render_mesh_to_image(gt_vertices, faces)
            img_pred = render_mesh_to_image(pred_vertices, faces)
            
            img_gt_bgr = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
            img_pred_bgr = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
            
            cv2.putText(img_gt_bgr, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img_pred_bgr, 'MLP Predicted', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            combined_img = np.hstack((img_gt_bgr, img_pred_bgr))
            
            img_save_path = os.path.join(output_dir, fname.replace('.json', '.png'))
            cv2.imwrite(img_save_path, combined_img)
            print(f" -> 照片保存成功！")
            
        except Exception as e:
            print(f"渲染照片失败: {e}")
        
    print(f"\n全部处理完成！文件已保存在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    RESULTS_DIRECTORY = "/home/abc/yg/LHM_Track/bs2flame_mapping/test_results" 
    FLAME_MODEL_DIR = "/home/abc/yg/models"
    
    visualize_results(RESULTS_DIRECTORY, FLAME_MODEL_DIR)