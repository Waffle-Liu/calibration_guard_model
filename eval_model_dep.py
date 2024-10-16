import os
import json
import argparse
import random
import csv
from tqdm import tqdm 
import numpy as np
import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, accuracy_score

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

from eval_utils import compute_results_classifier_hug
import calibration as cal

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




def seed_everything(seed: int):
    
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Eval LLM-based Guard Models.")
    parser.add_argument("--cls_path", type=str, default='meta-llama/LlamaGuard-7b',
                        help="The name or path of the guard model")
    parser.add_argument("--mode", type=str, default='prompt', choices=['prompt', 'prompt-response'],
                        help="The mode to evaluate, eg. prompt only, prompt-response")
    parser.add_argument("--dataset", type=str, default='harmbench',
                        help="The dataset to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--eval_full", type=bool, default=True,
                        help="Whether to eval F1, Precision, Recall, AUPRC")
    parser.add_argument("--num_tokens", type=int, default=512,
                        help="The number of tokens in response to evaluate")
    parser.add_argument("--cal_method", type=str, default='origin',
                        help="Calibration method")
    parser.add_argument("--ts", type=float, default=1.0,
                        help="Parameter for Temperature Scaling, default ts=1.0, no ts applied")
    
    args = parser.parse_args()
    return args

def calibrate_py(p_y, p_cf, mode='diagonal'):
    
    num_classes = p_y.shape[0]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == 'diagonal':
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
            cal_py = np.matmul(W, np.expand_dims(p_y, axis=-1)) + b 
        elif mode == 'identity':
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(np.log(p_cf), axis=-1)
            cal_py = np.matmul(W, np.expand_dims(np.log(p_y + 10e-6), axis=-1)) + b 
            cal_py = np.exp(cal_py)

    cal_py = cal_py/np.sum(cal_py)
    
    return cal_py


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    seed_everything(0)

    model_name_list = ['baichuan2', 'orca_2', 'gpt-4', 'starling', 'llama2', 'solar_10', 'vicuna', 'gpt-3.5', 'zephyr', 'qwen', 'claude-2', 'claude-instant', 'koala', 'openchat']

    if not os.path.exists("./data/classifier_val_sets/text_behaviors_val_set_model_dep.json"): 
        print("Prepare model-dependent prompt-repsonse dataset ...")
        behaviors_path = "./data/behavior_datasets/harmbench_behaviors_text_all.csv"
        with open(behaviors_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            behaviors = list(reader)
        behaviors = {b['BehaviorID']: b for b in behaviors}

        data_path = "./data/classifier_val_sets/text_behaviors_val_set.json" 
        with open(data_path, 'r') as f:
            val_set = json.load(f)
        
        completions = {}
        idx = 0
        model_stats = {}
        for bid, vals in val_set.items():
            for val in vals:
                completion = {}
                completion["behavior_id"] = bid
                if bid not in behaviors:
                    print(f"Behavior {bid} not found in behaviors file; skipping")
                    continue
                completion["generation"] = val['generation']
                exp = val["experiment"]
                if exp == 'default':
                    exp = val['model'].split('.json')[0]
                
                for model_name in model_name_list:
                    if model_name in exp:
                        response_model = model_name
                        break

                completion["response_model"] = response_model
                completion['prompt_gt'] = 1
                completion['response_gt'] = int((val['human_0'] + val['human_1'] + val['human_2']) >= 2)
        
                if response_model not in list(model_stats.keys()):
                    model_stats[response_model] = 1
                else:
                    model_stats[response_model] += 1

                completions[idx] = completion
                idx += 1
        print("Model-dependent dataset stats: ", model_stats)

        with open("./data/classifier_val_sets/text_behaviors_val_set_model_dep.json", 'w') as f_save:
            json.dump(completions, f_save, indent=4)
    
    if 'Aegis-AI-Content-Safety' in args.cls_path:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/LlamaGuard-7b')
        tokenizer.truncation_side="right"
        cls_m = AutoModelForCausalLM.from_pretrained('meta-llama/LlamaGuard-7b', torch_dtype=torch.bfloat16, device_map='cuda')
        cls = PeftModel.from_pretrained(cls_m, args.cls_path, device_map='cuda')

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.cls_path, trust_remote_code=True)
        tokenizer.truncation_side="right"
        cls = AutoModelForCausalLM.from_pretrained(args.cls_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cuda')


    mode = args.mode
    for response_model in model_name_list:
        if args.dataset == 'harmbench-adv-model':
            if mode == 'prompt-response':
                behaviors_path = "./data/behavior_datasets/harmbench_behaviors_text_all.csv"
                with open(behaviors_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    behaviors = list(reader)
                behaviors = {b['BehaviorID']: b for b in behaviors}

                data_path = "./data/classifier_val_sets/text_behaviors_val_set_model_dep.json" 
                with open(data_path, 'r') as f:
                    val_set = json.load(f)
                
                completions = {}
                model_stats = {}
                for idx, val in val_set.items():
                    if val['response_model'] == response_model: 
                        completion = {}
                        behavior_id = val['behavior_id']

                        behavior_dict = behaviors[behavior_id]
                        generation = val['generation']
                        tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
                        clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
                        completion['generation'] = clipped_generation
                        completion['prompt'] = behavior_dict['Behavior']
                        completion['prompt_gt'] = val['prompt_gt']
                        completion['response_gt'] = val['response_gt']
                        completions[idx] = completion      

        probs = []
        labels = []
        pred_labels = []
        all_probs = []

        cal_method = args.cal_method
        if cal_method == 'cc':
            data_cf = {
                "prompt": '',
                "generation": '',
                "prompt_gt": 0,
                "response_gt": 0
            }
            
            prob_cf, _, _, _ = compute_results_classifier_hug(data_cf, cls, tokenizer, args.cls_path, mode)
            prob_cf = np.array([prob_cf, 1-prob_cf])
            print(prob_cf)
            

        for idx, data in tqdm(completions.items()):
            prob, l, pred_l, _ = compute_results_classifier_hug(data, cls, tokenizer, args.cls_path, mode, t_s=args.ts)
        
            if cal_method == 'cc':
                prob = np.array([prob, 1-prob]) # unsafe, safe
                prob = calibrate_py(prob, prob_cf, mode='diagonal')
                probs.append(prob[0][0])
                pred_l = int(np.argmin(prob.reshape(-1)))
                
            else:
                probs.append(prob)

            labels.append(l)
            pred_labels.append(pred_l)
            all_probs.append([prob, 1-prob])
        
        if cal_method == 'bc':
            probs = []
            pred_labels = []
            all_probs = np.array(all_probs)
            batch_prob = 0

            batch_prob = np.mean(all_probs, axis=0)
            print(batch_prob)
            for i in range(all_probs.shape[0]):
                cal_prob  = calibrate_py(all_probs[i], batch_prob, mode='identity')
                probs.append(cal_prob[0][0])
                pred_l = int(np.argmin(cal_prob.reshape(-1)))
                pred_labels.append(pred_l)

        model_probs = np.array(probs)
        labels = np.array(labels)
        pred_labels = np.array(pred_labels)
        calibration_error = cal.get_calibration_error(model_probs, labels, debias=False)
        print("ECE: ", calibration_error)

        f1, precision, recall, accuracy, auprc = 0, 0, 0, 0, 0
        if args.eval_full:
            f1 = f1_score(labels, pred_labels)
            precision = precision_score(labels, pred_labels)
            recall = recall_score(labels, pred_labels)
            accuracy = accuracy_score(labels, pred_labels)
            print("Response Model: ", response_model)
            print("F1: ", f1)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("Accuracy: ", accuracy)

            precision_, recall_, thresholds = precision_recall_curve(labels, pred_labels)
            auprc = auc(recall_, precision_)
            print('AUPRC: ', auprc)        

        results = {
            'ece': calibration_error,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'acc': accuracy,
            'auprc': auprc
        }
        # Make dirs to output_file if not yet exist
        is_save = True
        if is_save == True:
            save_path = args.save_path + response_model + '.json'
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            with open(save_path, 'w') as file:
                json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()
