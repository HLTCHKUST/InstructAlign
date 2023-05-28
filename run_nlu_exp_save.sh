# bash run_nlu_prompt.sh ./save/monolingual/monolingual-bloomz-560m 2 &
# bash run_nlu_prompt.sh ./save/bilingual/bilingual-bloomz-560m 2 &
# bash run_nlu_prompt.sh ./save/translation/translation-bloomz-560m 3 &
# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m 3 &
# bash run_nlu_prompt.sh ./save/random/random-bloomz-560m 4

# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m_R-100 2 &
# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m_R-100/checkpoint-9740 2 &
# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m_R-100/checkpoint-19480 3 &
# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m_R-100/checkpoint-29220 3 &
# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m_R-100/checkpoint-38960 3
# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m_R-1000 2 &
# bash run_nlu_prompt.sh ./save/pair/pair-bloomz-560m_R-10000 2 &

bash run_nlu_prompt.sh ./save/pair/pair-bloomz-1b1 2 &
bash run_nlu_prompt.sh ./save/pair/pair-bloomz-1b1_R-100 3 &
bash run_nlu_prompt.sh ./save/pair/pair-bloomz-1b1_R-1000 4
bash run_nlu_prompt.sh ./save/pair/pair-bloomz-1b1_R-10000 5
