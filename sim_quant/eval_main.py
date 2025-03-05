import os
import time
import sys
import argparse

import torch

from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import BaseQuantizeConfig
from awq import AutoAWQForCausalLM

import lm_eval
import logging
from typing import Dict

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data
logger = None


def eval_task(model, tokenizer) -> Dict:
    
    task_fewshot_map = {
        "arc_easy": 25,
        "arc_challenge": 25,
        "hellaswag": 10,
        "mmlu": 5,
        "truthfulqa": 0,
        "winogrande": 5,
        "gsm8k": 5,
    }

    result_final = {}
    for _task, _num_fewshot in task_fewshot_map.items():
        start = time.time()

        result = lm_eval.simple_evaluate(
            model="hf",
            model_args={"pretrained":model, "tokenizer": tokenizer},
            tasks = [_task],
            num_fewshot = _num_fewshot,
            batch_size = 8,
            log_samples= False,
            device = f"cuda" if torch.cuda.is_available() else "cpu",
        )
        end = time.time()
        eval_time = end - start
    
        if _task == "arc_easy":
            result_final[_task] = round(result["results"]["arc_easy"]["acc_norm,none"], 4)
        elif _task == "arc_challenge":
            result_final[_task] = round(result["results"]["arc_challenge"]["acc_norm,none"], 4)
        elif _task == "hellaswag":
            result_final[_task] = round(result["results"]["hellaswag"]["acc_norm,none"], 4)
        elif _task == "mmlu":
            result_final[_task] = round(result["results"]["mmlu"]["acc,none"], 4)
        elif _task == "truthfulqa":
            result_final[_task] = round(result["results"]["truthfulqa_gen"]["bleu_acc,none"], 4)
        elif _task == "winogrande":
            result_final[_task] = round(result["results"]["winogrande"]["acc,none"], 4)
        elif _task == "gsm8k":
            result_final[_task] = round(result["results"]["gsm8k"]["exact_match,strict-match"], 4)
        
        logger.info(f"{_task} eval time = {eval_time}")
        logger.info(f"{_task} accuracy = {result}")

    return result_final
# def eval_task(model_path:str) -> tuple[dict, dict]:
    
#     task_fewshot_map = {
#         "arc_easy": 25,
#         "arc_challenge": 25,
#         "hellaswag": 10,
#         "mmlu": 5,
#         "truthfulqa": 0,
#         "winogrande": 5,
#         "gsm8k": 5,
#     }

#     for _task, _num_fewshot in task_fewshot_map.items():
#         start = time.time()

#         result = lm_eval.simple_evaluate(
#             model="hf",
#             model_args={"pretrained":model_path},
#             tasks = [_task],
#             num_fewshot = _num_fewshot,
#             batch_size = 8,
#             log_samples= False,
#             device = f"cuda" if torch.cuda.is_available() else "cpu",
#             limit = 2, ## for debugging
#         )
#         end = time.time()
#         eval_time = end - start
        
#         logger.info(f"{_task} eval time = {eval_time}")
#         logger.info(f"{_task} accuracy = {result}")


def setup_logger(filename: str) -> logging.Logger:
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="hqq quantized model evaluation for multiple tasks")
    
    # weight quant arguments
    parser.add_argument("--model_id", type=str, default="", help="huggingface model id")
    parser.add_argument("--quant_alg", type=str, default="awq", help="quantization algorithm (hqq or awq)")
    parser.add_argument("--nbits", type=int, default=4, help="weight bits")
    parser.add_argument("--group_size", type=int, default=128, help="group size")
    parser.add_argument("--axis", type=int, default=1, help="axis of per group quantization")
    parser.add_argument("--symm", action="store_true", help="symmetric quantization for 4bit (awq)")
    parser.add_argument("--shift_scale", action="store_true", help="shift scale for 4bit quant (awq)")
    parser.add_argument("--twosteps", action="store_true", help="first 8bit quant and then 4bit quant")
    
    # kv quant arguments
    parser.add_argument("--kv_quant", action="store_true", help="kv quantization")
    parser.add_argument("--kv_nbits", type=int, default=4, help="kv bits")
    parser.add_argument("--kv_group_size", type=int, default=128, help="kv group size")
    parser.add_argument("--kv_symm", action="store_true", help="kv symmetric quantization")
    parser.add_argument("--kv_shift_scale", action="store_true", help="shift scale for kv quant")
    parser.add_argument("--kv_quant_alg", type=str, default=None, help="kv quant algorithm (sageatten, smoothatten)")
    parser.add_argument("--k_only_sub", action="store_true", help="only k subtraction in sageatten")
    args = parser.parse_args()
    
    quant_name = f"{args.model_id}-{args.quant_alg}-n{args.nbits}-g{args.group_size}-a{args.axis}"
    if args.symm and args.quant_alg == "awq":
        quant_name += f"-symm"
    if args.shift_scale and args.quant_alg == "awq":
        quant_name += f"-shiftscale"
    if args.twosteps and args.quant_alg == "awq":
        quant_name += f"-twosteps"
    
    # load model
    model_path = os.path.join("./output_model", quant_name)
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = AutoAWQForCausalLM.from_quantized(model_path, device_map="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    
    eval_name = "eval"
    # kv quantization
    if args.kv_quant:
        eval_name += f"-kvquant-n{args.kv_nbits}-g{args.kv_group_size}"
        if args.kv_symm:
            eval_name += f"-symm"
        if args.kv_shift_scale:
            eval_name += f"-shiftscale"
        if args.kv_quant_alg is not None:
            eval_name += f"-{args.kv_quant_alg}"
        if args.k_only_sub and args.kv_quant_alg == "sageatten":
            eval_name += f"-Konlysub"
        for name, module in model.named_modules():
            if ("k_proj" in name or "v_proj" in name) and "WQLinear" in str(module):
                module.kv_quant = True
                module.kv_nbits = args.kv_nbits
                module.kv_group_size = args.kv_group_size
                module.kv_symm = args.kv_symm
                module.kv_shift_scale = args.kv_shift_scale
                if args.kv_quant_alg == "sageatten" and args.k_only_sub and "v_proj" in name:
                    module.kv_quant_alg = None
                else:
                    module.kv_quant_alg = args.kv_quant_alg
                module.name = name
                module.out_dir = model_path
    
    # print model
    print(model)
    
    # set logger
    global logger
    logger = setup_logger(f"{model_path}/{eval_name}.log")
    
    result = eval_task(model, tokenizer)
    logger.info(f"Final accuracy = {result}")

    #eval_task(model_path)

if __name__ == "__main__":
    sys.exit(main())
    