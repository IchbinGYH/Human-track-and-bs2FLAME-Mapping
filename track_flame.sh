#!/bin/bash

VIDEO_DIR="/home/abc/yg/LHM_Track/train_data"
OUTPUT_DIR="/home/abc/yg/LHM_Track/data/map_training"

for i in $(seq 2 15)
do
    VIDEO_PATH="${VIDEO_DIR}/${i}.mp4"

    echo "Processing ${VIDEO_PATH} ..."

    python track_video_yg.py \
        --video_path ${VIDEO_PATH} \
        --output_path ${OUTPUT_DIR}

    echo "Finished ${i}.mp4"
done

echo "All done!"