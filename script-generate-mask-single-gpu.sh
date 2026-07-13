CUDA_VISIBLE_DEVICES=0 python generate_mask.py \
    --video_path video_list/bear_g.mp4 \
    --output_path masks/bear_g_mask.mp4 \
    --text_prompt "large brown bear" \
    --backend sam \
    --box_threshold 0.25 \
    --text_threshold 0.25 \
    --max_boxes 3 \
    --dilate 9
