python train_gpt_xtts.py \
    --output_path ../checkpoints/xtts_egy_ar_v1 \
    --metadatas "../data/xtts_dataset/metadata_train.csv,../data/xtts_dataset/metadata_eval.csv,ar" \
    --num_epochs 30 \
    --batch_size 2 \
    --grad_acumm 8 \
    --lr 5e-6 \
    --save_step 5000 \
    --eval_step 10000 \
    --max_text_length 400
