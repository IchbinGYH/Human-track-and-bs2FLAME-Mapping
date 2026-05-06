import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 数据集加载与对齐
# ==========================================
class Blendshape2FlameDataset(Dataset):
    def __init__(self, root_dir):
        self.data_pairs = []
        self.bs_keys = None  # 记录并固定blendshape的键值顺序
        
        # 获取根目录下所有的子文件夹 (比如 1, 2, 3...)
        if not os.path.exists(root_dir):
            raise ValueError(f"你提供的根目录不存在，请检查路径: {root_dir}")
            
        subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        valid_folders = 0
        for folder_path in subdirs:
            flame_dir = os.path.join(folder_path, "flame_params")
            bs_dir = os.path.join(folder_path, "blendshape_params")
            
            # 如果这个子文件夹里没有我们要的这两个核心目录，就跳过
            if not os.path.exists(flame_dir) or not os.path.exists(bs_dir):
                continue
                
            valid_folders += 1
            
            # 获取两个文件夹中的文件名集合
            flame_files = set(os.listdir(flame_dir))
            bs_files = set(os.listdir(bs_dir))
            
            # 求交集，自动剔除丢失的数据
            valid_files = sorted(list(flame_files.intersection(bs_files)))
            
            for file_name in valid_files:
                if not file_name.endswith('.json'):
                    continue
                    
                bs_path = os.path.join(bs_dir, file_name)
                flame_path = os.path.join(flame_dir, file_name)
                
                # 读取JSON文件
                with open(bs_path, 'r') as f:
                    bs_data = json.load(f)
                with open(flame_path, 'r') as f:
                    flame_data = json.load(f)
                
                # 第一次读取时，确定并记录排序后的blendshape名称
                if self.bs_keys is None:
                    # 获取所有键名并排序
                    self.bs_keys = sorted([k for k in bs_data.keys()])
                    print("使用的 Blendshape Keys:", self.bs_keys)
                
                # 提取输入的 Blendshape 特征
                x_features = []
                for k in self.bs_keys:
                    # 如果键名是 _neutral，强制置为 0.0，否则读取实际值
                    if k == "_neutral":
                        x_features.append(0.0)
                    else:
                        x_features.append(bs_data[k])
                
                # 提取目标的 FLAME 特征
                y_exp = flame_data['expcode']              # 100维
                y_jaw = flame_data['posecode'][3:6]        # 3维 jaw pose
                y_eye = flame_data['eyecode']              # 6维
                
                self.data_pairs.append({
                    'x': torch.tensor(x_features, dtype=torch.float32),
                    'exp': torch.tensor(y_exp, dtype=torch.float32),
                    'jaw': torch.tensor(y_jaw, dtype=torch.float32),
                    'eye': torch.tensor(y_eye, dtype=torch.float32)
                })
        
        print(f"扫描完毕！共找到 {valid_folders} 个有效文件夹，成功匹配并加载了 {len(self.data_pairs)} 帧数据。")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        return item['x'], item['exp'], item['jaw'], item['eye']

# ==========================================
# 2. 定义 MLP 网络结构
# ==========================================
class MappingMLP(nn.Module):
    def __init__(self, input_dim=52, output_dim=100, hidden_layers=[256, 128]):
        super(MappingMLP, self).__init__()
        
        layers = []
        in_features = input_dim
        
        # 动态构建隐藏层
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # 添加BN层加速收敛
            layers.append(nn.ReLU())
            in_features = h_dim
            
        # 输出层 (不加激活函数，因为FLAME参数可正可负)
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. 训练主流程
# ==========================================
def train_mlp_models(root_dir, batch_size=128, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")
    
    # 准备数据
    dataset = Blendshape2FlameDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # 分别初始化三个MLP模型
    # 表情特征(100维)最复杂，给稍微宽一点的网络
    model_exp = MappingMLP(input_dim=52, output_dim=100, hidden_layers=[256, 128]).to(device)
    # 下巴特征(3维)和眼睛特征(6维)比较简单，网络可以稍微浅/窄一些防过拟合
    model_jaw = MappingMLP(input_dim=52, output_dim=3, hidden_layers=[128, 64]).to(device)
    model_eye = MappingMLP(input_dim=52, output_dim=6, hidden_layers=[128, 64]).to(device)
    
    # 定义损失函数 (MSE 均方误差对于回归任务最合适)
    criterion = nn.MSELoss()
    
    # 定义优化器 (Adam)
    optimizer_exp = optim.Adam(model_exp.parameters(), lr=lr)
    optimizer_jaw = optim.Adam(model_jaw.parameters(), lr=lr)
    optimizer_eye = optim.Adam(model_eye.parameters(), lr=lr)
    
    # 开始训练
    for epoch in range(epochs):
        model_exp.train()
        model_jaw.train()
        model_eye.train()
        
        total_loss_exp = 0.0
        total_loss_jaw = 0.0
        total_loss_eye = 0.0
        
        for x_bs, y_exp, y_jaw, y_eye in dataloader:
            x_bs = x_bs.to(device)
            y_exp, y_jaw, y_eye = y_exp.to(device), y_jaw.to(device), y_eye.to(device)
            
            # --- 训练 Expression 模型 ---
            optimizer_exp.zero_grad()
            pred_exp = model_exp(x_bs)
            loss_exp = criterion(pred_exp, y_exp)
            loss_exp.backward()
            optimizer_exp.step()
            total_loss_exp += loss_exp.item()
            
            # --- 训练 Jaw 模型 ---
            optimizer_jaw.zero_grad()
            pred_jaw = model_jaw(x_bs)
            loss_jaw = criterion(pred_jaw, y_jaw)
            loss_jaw.backward()
            optimizer_jaw.step()
            total_loss_jaw += loss_jaw.item()
            
            # --- 训练 Eye 模型 ---
            optimizer_eye.zero_grad()
            pred_eye = model_eye(x_bs)
            loss_eye = criterion(pred_eye, y_eye)
            loss_eye.backward()
            optimizer_eye.step()
            total_loss_eye += loss_eye.item()
            
        # 打印日志 (每 10 个 epoch 打印一次)
        if (epoch + 1) % 10 == 0:
            avg_loss_exp = total_loss_exp / len(dataloader)
            avg_loss_jaw = total_loss_jaw / len(dataloader)
            avg_loss_eye = total_loss_eye / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss_Exp: {avg_loss_exp:.6f} | Loss_Jaw: {avg_loss_jaw:.6f} | Loss_Eye: {avg_loss_eye:.6f}")
            
    # 训练结束后保存权重
    torch.save(model_exp.state_dict(), "mlp_exp_weights.pth")
    torch.save(model_jaw.state_dict(), "mlp_jaw_weights.pth")
    torch.save(model_eye.state_dict(), "mlp_eye_weights.pth")
    print("模型训练完成，权重已保存！")

# ==========================================
# 4. 执行代码
# ==========================================
if __name__ == "__main__":
    # 请将这里的 "your_dataset_folder" 替换为你实际包含 1~15 文件夹的大文件夹路径
    ROOT_DIRECTORY = "/home/abc/yg/LHM_Track/bs2flame_mapping/map_training_data" 
    
    # 建议先跑 100 个 epochs 观察收敛情况，如果还未收敛可适当调大 epochs
    train_mlp_models(ROOT_DIRECTORY, batch_size=128, epochs=145, lr=0.001)