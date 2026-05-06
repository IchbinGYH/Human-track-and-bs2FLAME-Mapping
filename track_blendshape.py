import os
import cv2
import json
import mediapipe as mp
import argparse

# ===================== 参数解析 =====================
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True)
parser.add_argument("--output_root", type=str, required=True)
parser.add_argument("--model_path", type=str, default="face_landmarker.task")

args = parser.parse_args()

video_path = args.video_path
base_output_dir = args.output_root
model_path = args.model_path

# ===================== 自动生成输出路径 =====================
video_name = os.path.splitext(os.path.basename(video_path))[0]

output_dir = os.path.join(
    base_output_dir,
    video_name,
    "blendshape_params"
)

os.makedirs(output_dir, exist_ok=True)

print("输入视频：", video_path)
print("输出路径：", output_dir)

# ===================== 初始化 MediaPipe =====================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.IMAGE
)

landmarker = FaceLandmarker.create_from_options(options)

# ===================== 读取视频 =====================
cap = cv2.VideoCapture(video_path)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    frame_data = {}

    if result.face_blendshapes:
        bs_list = result.face_blendshapes[0]

        for bs in bs_list:
            frame_data[bs.category_name] = float(bs.score)

        filename = f"{frame_idx:05d}.json"
        save_path = os.path.join(output_dir, filename)

        with open(save_path, "w") as f:
            json.dump(frame_data, f, indent=2)

    else:
        print("Frame", frame_idx, "没检测到人脸")

    if frame_idx % 50 == 0:
        print(f"Processed frame {frame_idx}")

cap.release()
print("Done!")