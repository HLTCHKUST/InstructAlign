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

###
# Monolingual
###

# Monolingual Rehearsal 100000
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
#     --output_dir ./save/monolingual/monolingual-bloomz-560m_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard
    
###
# Translation
###

# # Translation Rehearsal 100000
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
#     --output_dir ./save/translation/translation-bloomz-560m_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 2 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type translation \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard

# ###
# # Larger Model
# ###

###
# Monolingual
###

# # Monolingual Rehearsal 100000
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
#     --output_dir ./save/monolingual/monolingual-bloomz-1b1_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --resume_from_checkpoint ./save/monolingual/monolingual-bloomz-1b1_R-100000/checkpoint-40000 \
#     --fp16 \
#     --report_to tensorboard

###
# Translation
###

# Translation Rehearsal 100000
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
#     --output_dir ./save/translation/translation-bloomz-1b1_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type translation \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard
    
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

# ###
# # Bilingual
# ###

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
#     --output_dir ./save/bilingual/bilingual-bloomz-1b1_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
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
#     --output_dir ./save/bilingual/bilingual-bloomz-1b1_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard

# ###
# # XSS
# ###

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
#     --output_dir ./save/xss/xss-bloomz-1b1_R-10000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss \
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
#     --output_dir ./save/xss/xss-bloomz-1b1_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard


###
# ADDITIONAL EXPERIMENT FOR SEA WORKSHOP
###

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
#     --save_steps 100000 \
#     --save_total_limit 1 \
#     --output_dir ./save/multi/mlm-mt-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual,translation \
#     --fp16 \
#     --report_to tensorboard &
    
# CUDA_VISIBLE_DEVICES=1 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
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
#     --output_dir ./save/multi/mlm-tlm-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual,bilingual \
#     --fp16 \
#     --report_to tensorboard &

# CUDA_VISIBLE_DEVICES=2 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
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
#     --output_dir ./save/multi/mlm-xss-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual,xss \
#     --fp16 \
#     --report_to tensorboard &
    
# wait
# wait
# wait

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
#     --save_steps 100000 \
#     --save_total_limit 1 \
#     --output_dir ./save/multi/xss-mt-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type xss,translation \
#     --fp16 \
#     --report_to tensorboard &

# CUDA_VISIBLE_DEVICES=1 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
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
#     --output_dir ./save/multi/tlm-xss-mt-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual,xss,translation \
#     --fp16 \
#     --report_to tensorboard &

###
# Longer Multi Objectives    
###
# CUDA_VISIBLE_DEVICES=1 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 100000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/multi/tlm-xss-bloomz-560m-S-100k \
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

# CUDA_VISIBLE_DEVICES=2 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --max_steps 150000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 200000 \
#     --save_total_limit 1 \
#     --output_dir ./save/multi/tlm-xss-mt-bloomz-560m-longer \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual,xss,translation \
#     --fp16 \
#     --report_to tensorboard &
    
# wait
# wait
# wait

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
#     --save_steps 100000 \
#     --save_total_limit 1 \
#     --output_dir ./save/multi/tlm-mt-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual,translation \
#     --fp16 \
#     --report_to tensorboard &
    
# CUDA_VISIBLE_DEVICES=1 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
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
#     --output_dir ./save/multi/mlm-tlm-mt-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual,bilingual,translation \
#     --fp16 \
#     --report_to tensorboard &

# CUDA_VISIBLE_DEVICES=2 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
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
#     --output_dir ./save/multi/mlm-tlm-xss-bloomz-560m \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 16 \
#     --dataloader_num_workers 16 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type monolingual,bilingual,xss \
#     --fp16 \
#     --report_to tensorboard &
    
# wait
# wait
# wait

CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-560m \
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
    --output_dir ./save/multi/tlm-bloomz-560m \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --augmentation_type bilingual \
    --fp16 \
    --report_to tensorboard &

CUDA_VISIBLE_DEVICES=1 python run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-560m \
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
    --output_dir ./save/multi/xss-bloomz-560m \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --augmentation_type xss \
    --fp16 \
    --report_to tensorboard &

CUDA_VISIBLE_DEVICES=2 python run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-560m \
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
    --output_dir ./save/multi/mlm-bloomz-560m \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --augmentation_type monolingual \
    --fp16 \
    --report_to tensorboard &
    
CUDA_VISIBLE_DEVICES=3 python run_t2t_finetuning.py \
    --model_name_or_path bigscience/bloomz-560m \
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
    --output_dir ./save/multi/mt-bloomz-560m \
    --learning_rate 1e-5 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --augmentation_type translation \
    --fp16 \
    --report_to tensorboard &

wait
wait
wait
wait

###
# ADDITIONAL FOR REBUTTAL
###

# # Smaller Data Size no Rehearsal
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --num_train_ratio 0.5 \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual-few/bloomz-560m-500 \
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
    
# CUDA_VISIBLE_DEVICES=1 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --num_train_ratio 0.25 \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual-few/bloomz-560m-1000 \
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

# # Smaller Data Size R-100000
# CUDA_VISIBLE_DEVICES=0 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --num_train_ratio 0.5 \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual-few/bloomz-560m-500_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --augmentation_type bilingual \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --fp16 \
#     --report_to tensorboard &
    
# CUDA_VISIBLE_DEVICES=1 python run_t2t_finetuning.py \
#     --model_name_or_path bigscience/bloomz-560m \
#     --do_train \
#     --do_eval \
#     --num_train_ratio 0.25 \
#     --max_steps 50000 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 1000 \
#     --save_strategy steps \
#     --save_steps 10000 \
#     --save_total_limit 1 \
#     --output_dir ./save/bilingual-few/bloomz-560m-1000_R-100000 \
#     --learning_rate 1e-5 \
#     --preprocessing_num_workers 32 \
#     --dataloader_num_workers 32 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing \
#     --overwrite_output_dir \
#     --continual_type rehearsal \
#     --continual_size 100000 \
#     --augmentation_type bilingual \
#     --fp16 \
#     --report_to tensorboard &