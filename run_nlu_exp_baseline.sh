bash run_nlu_prompt.sh bigscience/bloom-560m 1 &
bash run_nlu_prompt.sh bigscience/bloom-1b1 1 &
# bash run_nlu_prompt.sh bigscience/bloom-1b7 1 &
bash run_nlu_prompt.sh bigscience/bloom-3b 1 &

wait
wait
wait

bash run_nlu_prompt.sh bigscience/bloomz-560m 1 &
bash run_nlu_prompt.sh bigscience/bloomz-1b1 1 &
# bash run_nlu_prompt.sh bigscience/bloomz-1b7 3 &
bash run_nlu_prompt.sh bigscience/bloomz-3b 1