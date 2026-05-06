import os
import json
import torch
import torch.nn as nn
import glob

# ==========================================
# 1. MLP 网络结构 (保持不变，与训练完全对齐)
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
# 2. 批量推理与生成 FLAME JSON
# ==========================================
def batch_inference_bs_to_flame(input_bs_dir, output_flame_dir, weight_exp, weight_jaw):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用计算设备: {device}")
    
    print("⏳ 正在加载预训练模型权重 (Exp & Jaw)...")
    model_exp = MappingMLP(input_dim=52, output_dim=100, hidden_layers=[256, 128]).to(device)
    model_jaw = MappingMLP(input_dim=52, output_dim=3, hidden_layers=[128, 64]).to(device)
    
    model_exp.load_state_dict(torch.load(weight_exp, map_location=device))
    model_jaw.load_state_dict(torch.load(weight_jaw, map_location=device))
    
    # 开启评估模式
    model_exp.eval()
    model_jaw.eval()

    if not os.path.exists(input_bs_dir):
        raise ValueError(f"❌ 提供的输入目录不存在: {input_bs_dir}")
        
    os.makedirs(output_flame_dir, exist_ok=True)
    
    json_files = sorted(glob.glob(os.path.join(input_bs_dir, "*.json")))
    if not json_files:
        print(f"❌ 在 {input_bs_dir} 中没有找到任何 .json 文件！")
        return
        
    print(f"🔍 找到 {len(json_files)} 个 Blendshape 文件，开始逐帧转换...")

    # 🌟 完美的 52 维列表（严格对齐训练时的键名和顺序）
    TRAINING_KEYS_SORTED = [
        "_neutral",
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
        "noseSneerLeft", "noseSneerRight"
    ]

    # ---------------- 3. 开始逐帧推理 ----------------
    with torch.no_grad():
        for idx, json_path in enumerate(json_files):
            with open(json_path, 'r') as f:
                bs_data = json.load(f)
            
            # 💡 核心修复 1：绝对不篡改输入！老老实实把 52 维真实数据交给 MLP
            x_feature = [bs_data.get(k, 0.0) for k in TRAINING_KEYS_SORTED]
            x_tensor = torch.tensor([x_feature], dtype=torch.float32).to(device)
            
            # 推理得到最真实的表情和下巴
            pred_exp = model_exp(x_tensor).squeeze(0).cpu().tolist() 
            pred_jaw = model_jaw(x_tensor).squeeze(0).cpu().tolist() 
            
            # 💡 核心修复 2：绝对不干预输出尺度！原汁原味组装
            # 强行屏蔽表情！只看下巴！
            flame_dict = {
                "shapecode": [0.0] * 100,               # 基础体型设为0
                "expcode": pred_exp,                    # MLP 预测的表情（已过滤掉眼睛乱动）
                "posecode": [0.0, 0.0, 0.0] + pred_jaw, # 头部固定 + MLP 预测的下巴
                "neckcode": [0.0, 0.0, 0.0],            # 脖子固定
                "eyecode": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 眼球强制平视前方
            }
            
            file_name = os.path.basename(json_path)
            output_path = os.path.join(output_flame_dir, file_name)
            
            with open(output_path, 'w') as out_f:
                json.dump(flame_dict, out_f, indent=2)
                
            if (idx + 1) % 100 == 0:
                print(f"⏳ 已处理 {idx + 1} / {len(json_files)} 帧...")

    print(f"\n✅ 全部转换完成！输入输出已完全还原为模型原貌。")
    print(f"📁 输出保存在: {os.path.abspath(output_flame_dir)}")

if __name__ == "__main__":
    INPUT_BS_DIR = "/home/abc/yg/LHM_Track/bs2flame_mapping/map_test_data/3/blendshape_param"
    OUTPUT_FLAME_DIR = "/home/abc/yg/LHM_Track/bs2flame_mapping/map_test_data/3/flame_param"  
    
    WEIGHT_EXP = "mlp_exp_weights.pth"
    WEIGHT_JAW = "mlp_jaw_weights.pth"
    
    batch_inference_bs_to_flame(
        input_bs_dir=INPUT_BS_DIR, 
        output_flame_dir=OUTPUT_FLAME_DIR, 
        weight_exp=WEIGHT_EXP, 
        weight_jaw=WEIGHT_JAW
    )