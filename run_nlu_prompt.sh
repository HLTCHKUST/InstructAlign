# bigscience/mt0-base
# bigscience/mt0-large
# bigscience/bloomz-560m
# bigscience/bloomz-1b1

echo "$1 $2"
CUDA_VISIBLE_DEVICES=$2 python main_nlu_prompt.py ID $1
CUDA_VISIBLE_DEVICES=$2 python main_nlu_prompt.py ID2 $1
CUDA_VISIBLE_DEVICES=$2 python main_nlu_prompt.py ID3 $1
CUDA_VISIBLE_DEVICES=$2 python main_nlu_prompt.py EN $1
CUDA_VISIBLE_DEVICES=$2 python main_nlu_prompt.py EN2 $1
CUDA_VISIBLE_DEVICES=$2 python main_nlu_prompt.py EN3 $1