#!/bin/bash

VIDEO_DIR="/home/abc/yg/LHM_Track/train_data"
OUTPUT_ROOT="/home/abc/yg/LHM_Track/data/map_training"

SCRIPT="track_blendshape.py"

for i in $(seq 1 15)
do
    VIDEO_PATH="${VIDEO_DIR}/${i}.mp4"

    if [ ! -f "$VIDEO_PATH" ]; then
        echo "⚠️ ${VIDEO_PATH} 不存在，跳过"
        continue
    fi

    echo "Processing ${VIDEO_PATH}"

    python $SCRIPT \
        --video_path $VIDEO_PATH \
        --output_root $OUTPUT_ROOT

    echo "Finished ${i}.mp4"
done

echo "All done!"