import os
import cv2
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 🌟 标准 52 维列表
STANDARD_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
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

def process_single_video(video_path, output_dir, model_asset_path):
    """提取单个视频的 Blendshape，严格按照物理帧号保存，遇缺则空"""
    
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    
    with vision.FaceLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None: fps = 30.0 
        
        video_frame_idx = 0 # 记录视频真实的物理帧号 (从 0 开始)
        saved_count = 0     # 仅仅用来统计成功保存了多少文件
        
        os.makedirs(output_dir, exist_ok=True)
        
        last_timestamp_ms = -1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            # 手动计算时间戳并加上单调递增锁
            timestamp_ms = int(video_frame_idx * 1000 / fps)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms 

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 运行检测
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)

            # 只有检测到人脸时，才构建字典并保存
            if detection_result.face_blendshapes:
                frame_dict = {}
                blendshapes = detection_result.face_blendshapes[0]
                for category in blendshapes:
                    frame_dict[category.category_name] = category.score
                    
                if "_neutral" not in frame_dict:
                    frame_dict = {"_neutral": 0.0, **frame_dict}

                # 💡 核心修正：使用物理帧号 `video_frame_idx` 作为文件名！
                # 这样如果第2帧没检测到，就不会生成 00002.json，下一个文件直接是 00003.json
                file_name = f"{video_frame_idx:05d}.json"
                with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
                    json.dump(frame_dict, f, indent=2)
                    
                saved_count += 1
            else:
                # 没检测到人脸时，什么都不存，直接跳过
                print(f"⚠️ 视频 {os.path.basename(video_path)} 第 {video_frame_idx:05d} 帧未检测到人脸，将产生标号空缺。")
                
            video_frame_idx += 1 # 物理帧号始终正常推进

        cap.release()
        print(f"✅ 完成: {os.path.basename(video_path)} -> 视频共 {video_frame_idx} 帧，成功保存 {saved_count} 帧。")

def batch_extract_videos(root_dir, model_asset_path):
    if not os.path.exists(model_asset_path):
        raise FileNotFoundError(f"找不到模型文件 {model_asset_path}！请先下载 face_landmarker.task。")
        
    print("✅ 开始扫描文件夹...\n")
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for folder_path in subdirs:
        folder_name = os.path.basename(folder_path)
        video_path = os.path.join(folder_path, f"{folder_name}.mp4")
        
        if os.path.exists(video_path):
            output_dir = os.path.join(folder_path, "blendshape_params")
            process_single_video(video_path, output_dir, model_asset_path)
        else:
            print(f"⏭️ 跳过: {folder_name} 中没有找到 {folder_name}.mp4")

    print("\n🎉 所有视频的 Blendshape 提取完毕！")

# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 填写你的大文件夹路径
    ROOT_DATA_DIR = "/home/abc/yg/LHM_Track/bs2flame_mapping/LAM_data/export"
    
    # MediaPipe 模型文件路径
    MODEL_PATH = "/home/abc/yg/LHM_Track/face_landmarker.task"
    
    batch_extract_videos(ROOT_DATA_DIR, MODEL_PATH)