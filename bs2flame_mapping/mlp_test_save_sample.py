import os
import json
import torch
import torch.nn as nn
import random
import shutil

# --- 网络结构 (保持一致) ---
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
    def forward(self, x): return self.network(x)

def test_and_save_results(root_dir, weights, num_samples=10, save_dir="/home/abc/yg/LHM_Track/bs2flame_mapping/test_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    m_exp = MappingMLP(52, 100, [256, 128]).to(device)
    m_jaw = MappingMLP(52, 3, [128, 64]).to(device)
    m_eye = MappingMLP(52, 6, [128, 64]).to(device)
    m_exp.load_state_dict(torch.load(weights['exp'], map_location=device))
    m_jaw.load_state_dict(torch.load(weights['jaw'], map_location=device))
    m_eye.load_state_dict(torch.load(weights['eye'], map_location=device))
    m_exp.eval(); m_jaw.eval(); m_eye.eval()

    # 2. 准备保存目录
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # 3. 数据扫描 (简化逻辑)
    all_data = []
    bs_keys = None
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for folder in subdirs:
        f_dir, b_dir = os.path.join(folder, "flame_params"), os.path.join(folder, "blendshape_params")
        if not (os.path.exists(f_dir) and os.path.exists(b_dir)): continue
        valid = sorted(list(set(os.listdir(f_dir)) & set(os.listdir(b_dir))))
        for f in valid:
            if f.endswith('.json'): all_data.append((os.path.join(b_dir, f), os.path.join(f_dir, f), f))

    # 4. 抽取样本并保存
    samples = random.sample(all_data, min(num_samples, len(all_data)))
    print(f"正在保存 {len(samples)} 个测试样本到 {save_dir}...")

    with torch.no_grad():
        for bs_p, fl_p, fname in samples:
            with open(bs_p, 'r') as f: bs_in = json.load(f)
            with open(fl_p, 'r') as f: fl_gt = json.load(f)
            if bs_keys is None: bs_keys = sorted(bs_in.keys())
            
            x = torch.tensor([[bs_in[k] for k in bs_keys]], dtype=torch.float32).to(device)
            p_exp = m_exp(x).cpu().numpy()[0].tolist()
            p_jaw = m_jaw(x).cpu().numpy()[0].tolist()
            p_eye = m_eye(x).cpu().numpy()[0].tolist()
            print("p_exp", p_exp)
            print("p_exp_gt", fl_gt['expcode'])
            print("p_jaw", p_jaw)
            print("p_jaw_gt", fl_gt['posecode'][3:6])
            print("p_eye", p_eye)
            print("p_eye_gt", fl_gt['eyecode'])

            # 构造保存结构
            result = {
                "source_file": bs_p,
                "input_blendshapes": bs_in, # 原始52维输入
                "ground_truth_flame": {
                    "exp": fl_gt['expcode'],
                    "jaw": fl_gt['posecode'][3:6],
                    "eye": fl_gt['eyecode']
                },
                "predicted_flame": {
                    "exp": p_exp,
                    "jaw": p_jaw,
                    "eye": p_eye
                }
            }
            with open(os.path.join(save_dir, f"res_{fname}"), 'w') as f:
                json.dump(result, f, indent=2)
    print("保存完成。")

if __name__ == "__main__":
    ROOT = r"/home/abc/yg/LHM_Track/bs2flame_mapping/map_training"
    W = {'exp': "mlp_exp_weights.pth", 'jaw': "mlp_jaw_weights.pth", 'eye': "mlp_eye_weights.pth"}
    test_and_save_results(ROOT, W, num_samples=5) # 抽5个结果