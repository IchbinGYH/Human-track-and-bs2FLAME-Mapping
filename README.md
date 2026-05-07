# cuda 12.1
sh ./install_cu121.sh

# cuda 11.8
sh ./install_cu118.sh
```
The installation has been tested with python3.10, CUDA 12.1 or CUDA 11.8.



### Download Prior Model Weights 
Download basic model weights. If you've downloaded them in LHM repo, you can copy/link the `pretrained_models` folder into this repo.
```bash
# Download prior model weights
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar 
tar -xvf LHM_prior_model.tar 
```

Download extra model weights.
```bash
# Download extra model weights
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_track_model.tar 
tar -xvf LHM_track_model.tar 
```


### 🏃 Inference Pipeline
```bash
python track_video.py --video_path ${VIDEO_PATH} --output_path ${OUTPUT_PATH}
```

### 🏃 Inference Mediapipe
```bash
python track_blendshape.py --video_path ${VIDEO_PATH} --output_path ${OUTPUT_PATH}
```

### 🏃 Inference More Than 2 videos' Mediapipe
```bash
bash ./track_blendshape.sh
```


