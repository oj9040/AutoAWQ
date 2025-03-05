
MODELS=("meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct")
for MODEL in "${MODELS[@]}"
do
    #CUDA_VISIBLE_DEVICES=2 python3 quant_main.py --model_id ${MODEL} --quant_alg awq --nbits 4 --group_size 128 --axis 1 &&
    #CUDA_VISIBLE_DEVICES=2 python3 eval_main.py --model_id ${MODEL} --quant_alg awq --nbits 4 --group_size 128 --axis 1 &&
    #CUDA_VISIBLE_DEVICES=2 python3 eval_main.py --model_id ${MODEL} --quant_alg awq --nbits 4 --group_size 128 --axis 1 --kv_quant --kv_nbits 4 --kv_group_size 128 &&
    CUDA_VISIBLE_DEVICES=2 python3 eval_main.py --model_id ${MODEL} --quant_alg awq --nbits 4 --group_size 128 --axis 1 --kv_quant --kv_nbits 4 --kv_group_size 128 --kv_quant_alg sageatten &&
    CUDA_VISIBLE_DEVICES=2 python3 eval_main.py --model_id ${MODEL} --quant_alg awq --nbits 4 --group_size 64 --axis 1 --kv_quant --kv_nbits 4 --kv_group_size 64 --kv_quant_alg sageatten
done