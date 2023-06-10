# Monolingual
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/monolingual/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual \
#     --fp16 \
#     --report_to tensorboard &

# # Bilingual
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
#     --fp16 \
#     --report_to tensorboard &
    
# # Translation
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/translation/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type translation \
#     --fp16 \
#     --report_to tensorboard &
    
# # XSS
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/xss/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss \
#     --fp16 \
#     --report_to tensorboard &

# # Bilingual XSS
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual-xss/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual-xss \
#     --fp16 \
#     --report_to tensorboard &

# # Pair
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --fp16 \
#     --report_to tensorboard &

# # Pair + XSS
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair-xss/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair-xss \
#     --fp16 \
#     --report_to tensorboard &
    
# # Random
# CUDA_VISIBLE_DEVICES=4 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/random/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type random \
#     --fp16 \
#     --report_to tensorboard &
    
# # Random + XSS
# CUDA_VISIBLE_DEVICES=5 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/random-xss/bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type random-xss \
#     --fp16 \
#     --report_to tensorboard &

# ###
# # Rehearsal 
# ###

# # Pair Rehearsal 100
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-560m_R-100 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 100 \
#     --fp16 \
#     --report_to tensorboard 

# # Pair Rehearsal 1000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-560m_R-1000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 1000 \
#     --fp16 \
#     --report_to tensorboard

# # Pair Rehearsal 10000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-560m_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 10000 \
#     --fp16 \
#     --report_to tensorboard

# # Pair Rehearsal 100000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-560m_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard

# # Bilingual Rehearsal 100
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual/bloomz-560m_R-100 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
#     --continual_type rehearsal \
#     --continual_size 100 \
#     --fp16 \
#     --report_to tensorboard

# # Bilingual Rehearsal 1000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual/bloomz-560m_R-1000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
#     --continual_type rehearsal \
#     --continual_size 1000 \
#     --fp16 \
#     --report_to tensorboard

# # Bilingual Rehearsal 10000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual/bloomz-560m_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
#     --continual_type rehearsal \
#     --continual_size 10000 \
#     --fp16 \
#     --report_to tensorboard

# # Bilingual Rehearsal 100000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual/bloomz-560m_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard


# # XSS Rehearsal 100
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/xss/bloomz-560m_R-100 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss \
#     --continual_type rehearsal \
#     --continual_size 100 \
#     --fp16 \
#     --report_to tensorboard

# # XSS Rehearsal 1000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/xss/bloomz-560m_R-1000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss \
#     --continual_type rehearsal \
#     --continual_size 1000 \
#     --fp16 \
#     --report_to tensorboard

# # XSS Rehearsal 10000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/xss/bloomz-560m_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss \
#     --continual_type rehearsal \
#     --continual_size 10000 \
#     --fp16 \
#     --report_to tensorboard

# # XSS Rehearsal 100000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/xss/bloomz-560m_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard

# ###
# # Larger Model
# ###

###
# Pair
###

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b1 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b1 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --fp16 \
#     --report_to tensorboard

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b1 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b1_R-100 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 100 \
#     --fp16 \
#     --report_to tensorboard 

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b1 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b1_R-1000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 1000 \
#     --fp16 \
#     --report_to tensorboard 

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b1 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b1_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 10000 \
#     --fp16 \
#     --report_to tensorboard 

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b1 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 100000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b1_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard

###
# Bilingual
###

CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-1b1 \
    --do_train \
    --do_eval \
    --max_steps 50000 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 100000 \
    --save_total_limit 1 \
    --output_dir ./save/bilingual/bilingual-bloomz-1b1_R-100000 \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --augmentation_type bilingual \
    --continual_type rehearsal \
    --continual_size 100000 \
    --fp16 \
    --report_to tensorboard

###
# XSS
###

CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-1b1 \
    --do_train \
    --do_eval \
    --max_steps 50000 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 100000 \
    --save_total_limit 1 \
    --output_dir ./save/xss/xss-bloomz-1b1_R-100000 \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --augmentation_type xss \
    --continual_type rehearsal \
    --continual_size 100000 \
    --fp16 \
    --report_to tensorboard

###
# Even Larger Model
###
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b7 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b7 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --fp16 \
#     --report_to tensorboard

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b7 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b7_R-100 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 8 \
#     --dataloader_num_workers 8 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 100 \
#     --fp16 \
#     --report_to tensorboard 

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b7 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b7_R-1000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 8 \
#     --dataloader_num_workers 8 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 1000 \
#     --fp16 \
#     --report_to tensorboard 

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b7 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b7_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 8 \
#     --dataloader_num_workers 8 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 10000 \
#     --fp16 \
#     --report_to tensorboard 

# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-1b7 \
#     --do_train \
#     --do_eval \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 100000 \
#     --save_total_limit 1 \
#     --output_dir ./save/pair/bloomz-1b7_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 8 \
#     --dataloader_num_workers 8 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type pair \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard 