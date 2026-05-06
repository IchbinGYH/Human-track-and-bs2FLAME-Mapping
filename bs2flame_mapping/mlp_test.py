import os
import json
import torch
import torch.nn as nn
import random

# ==========================================
# 1. 重新定义 MLP 网络结构 (必须和训练时完全一致)
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
# 2. 测试与评估主逻辑
# ==========================================
def test_mlp_models(root_dir, weight_exp, weight_jaw, weight_eye):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")
    
    # ---------------- 1. 加载模型与权重 ----------------
    print("正在加载模型权重...")
    model_exp = MappingMLP(input_dim=52, output_dim=100, hidden_layers=[256, 128]).to(device)
    model_jaw = MappingMLP(input_dim=52, output_dim=3, hidden_layers=[128, 64]).to(device)
    model_eye = MappingMLP(input_dim=52, output_dim=6, hidden_layers=[128, 64]).to(device)
    
    # map_location 确保即使你换了没GPU的机器也能跑
    model_exp.load_state_dict(torch.load(weight_exp, map_location=device))
    model_jaw.load_state_dict(torch.load(weight_jaw, map_location=device))
    model_eye.load_state_dict(torch.load(weight_eye, map_location=device))
    
    # 开启评估模式 (非常重要！会冻结 Dropout 和 BatchNorm)
    model_exp.eval()
    model_jaw.eval()
    model_eye.eval()

    # ---------------- 2. 扫描并加载测试数据 ----------------
    if not os.path.exists(root_dir):
        raise ValueError(f"提供的根目录不存在: {root_dir}")
        
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    bs_keys = None
    all_x, all_y_exp, all_y_jaw, all_y_eye = [], [], [], []
    sample_files = [] # 记录文件名用于展示
    
    for folder_path in subdirs:
        flame_dir = os.path.join(folder_path, "flame_params")
        bs_dir = os.path.join(folder_path, "blendshape_params")
        
        if not os.path.exists(flame_dir) or not os.path.exists(bs_dir):
            continue
            
        flame_files = set(os.listdir(flame_dir))
        bs_files = set(os.listdir(bs_dir))
        valid_files = sorted(list(flame_files.intersection(bs_files)))
        
        for file_name in valid_files:
            if not file_name.endswith('.json'): continue
                
            with open(os.path.join(bs_dir, file_name), 'r') as f:
                bs_data = json.load(f)
            with open(os.path.join(flame_dir, file_name), 'r') as f:
                flame_data = json.load(f)
            
            if bs_keys is None:
                bs_keys = sorted([k for k in bs_data.keys()])
            
            # 提取特征
            all_x.append([bs_data[k] for k in bs_keys])
            all_y_exp.append(flame_data['expcode'])
            all_y_jaw.append(flame_data['posecode'][3:6])
            all_y_eye.append(flame_data['eyecode'])
            sample_files.append(os.path.join(folder_path, file_name))
            
    if len(all_x) == 0:
        print("未找到任何配对的有效数据，请检查路径。")
        return

    # 转换为 Tensor
    x_tensor = torch.tensor(all_x, dtype=torch.float32).to(device)
    y_exp_tensor = torch.tensor(all_y_exp, dtype=torch.float32).to(device)
    y_jaw_tensor = torch.tensor(all_y_jaw, dtype=torch.float32).to(device)
    y_eye_tensor = torch.tensor(all_y_eye, dtype=torch.float32).to(device)

    print(f"数据加载完成！共读取 {len(all_x)} 帧配对数据。")

    # ---------------- 3. 计算整体评估指标 (不计算梯度) ----------------
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss() # MAE (平均绝对误差) 更加直观

    with torch.no_grad():
        pred_exp = model_exp(x_tensor)
        pred_jaw = model_jaw(x_tensor)
        pred_eye = model_eye(x_tensor)
        
        mse_exp = criterion_mse(pred_exp, y_exp_tensor).item()
        mae_exp = criterion_mae(pred_exp, y_exp_tensor).item()
        
        mse_jaw = criterion_mse(pred_jaw, y_jaw_tensor).item()
        mae_jaw = criterion_mae(pred_jaw, y_jaw_tensor).item()
        
        mse_eye = criterion_mse(pred_eye, y_eye_tensor).item()
        mae_eye = criterion_mae(pred_eye, y_eye_tensor).item()

    print("\n" + "="*40)
    print("模型整体评估结果 (整个测试集):")
    print(f"[Expression (100维)] -> MSE: {mse_exp:.6f} | MAE: {mae_exp:.6f}")
    print(f"[Jaw Pose   (3维 )] -> MSE: {mse_jaw:.6f} | MAE: {mae_jaw:.6f}")
    print(f"[Eye Pose   (6维 )] -> MSE: {mse_eye:.6f} | MAE: {mae_eye:.6f}")
    print("="*40)

    # ---------------- 4. 随机抽查一帧进行肉眼比对 ----------------
    idx = random.randint(0, len(all_x) - 1)
    print(f"\n[随机抽查验证] 文件来源: {sample_files[idx]}")
    
    # 提取那特定一帧的数据，转成 list 方便打印
    gt_jaw = y_jaw_tensor[idx].cpu().tolist()
    p_jaw = pred_jaw[idx].cpu().tolist()
    
    gt_eye = y_eye_tensor[idx].cpu().tolist()
    p_eye = pred_eye[idx].cpu().tolist()
    
    # 100维太长了，我们只打印前 10 维感受一下
    gt_exp = y_exp_tensor[idx].cpu().tolist()[:10] 
    p_exp = pred_exp[idx].cpu().tolist()[:10]

    print("\n--- Jaw Pose (3维) 对比 ---")
    print(f"真实值: {[round(v, 4) for v in gt_jaw]}")
    print(f"预测值: {[round(v, 4) for v in p_jaw]}")
    
    print("\n--- Eye Pose (6维) 对比 ---")
    print(f"真实值: {[round(v, 4) for v in gt_eye]}")
    print(f"预测值: {[round(v, 4) for v in p_eye]}")
    
    print("\n--- Expression (仅展示前10维) 对比 ---")
    print(f"真实值: {[round(v, 4) for v in gt_exp]}")
    print(f"预测值: {[round(v, 4) for v in p_exp]}")
    print("\n测试结束。")


# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    # 1. 填写你的数据大文件路径
    ROOT_DIRECTORY = "/home/abc/yg/LHM_Track/bs2flame_mapping/map_training" 
    
    # 2. 填写刚才训练出来的 3 个权重文件的相对路径（如果在同一目录下就直接写名字）
    WEIGHT_EXP = "mlp_exp_weights.pth"
    WEIGHT_JAW = "mlp_jaw_weights.pth"
    WEIGHT_EYE = "mlp_eye_weights.pth"
    
    test_mlp_models(ROOT_DIRECTORY, WEIGHT_EXP, WEIGHT_JAW, WEIGHT_EYE)