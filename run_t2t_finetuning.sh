# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --num_train_epochs 50 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy epoch \
#     --eval_steps 1 \
#     --save_strategy epoch \
#     --save_steps 10 \
#     --save_total_limit 5 \
#     --output_dir ./save/monolingual/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 186 \
#     --dataloader_num_workers 8 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --overwrite_output_dir \
#     --augmentation_type monolingual \
#     --fp16 \
#     --sharded_ddp zero_dp_3

TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-560m \
    --do_train \
    --num_train_epochs 50 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --eval_steps 1 \
    --save_strategy epoch \
    --save_steps 10 \
    --save_total_limit 5 \
    --output_dir ./save/translation/bloomz-560m \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --augmentation_type translation \
    --fp16 \
    --sharded_ddp zero_dp_3

TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-560m \
    --do_train \
    --num_train_epochs 50 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --eval_steps 1 \
    --save_strategy epoch \
    --save_steps 10 \
    --save_total_limit 5 \
    --output_dir ./save/bilingual/bloomz-560m \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --augmentation_type bilingual \
    --fp16 \
    --sharded_ddp zero_dp_3    
