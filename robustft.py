import random
import pandas as pd
import os
import copy
import numpy as np
from common import *
from eval import Eval
import threading
from config import *
from rag import NearestReference
import argparse


def calculate_entropy(probs):
    prob_list = np.array(probs)
    entropy = - np.sum(prob_list) / len(prob_list)
    return entropy

# Add argument parser
parser = argparse.ArgumentParser(description='Noisy Free Fine-tuning')
parser.add_argument('--task', type=str, default='mmlu', help='Task name')
parser.add_argument('--model', type=str, default='llama3.1-8b', help='Model name')
parser.add_argument('--noise_ratio', type=int, default=50, help='Noise ratio')
parser.add_argument('--base_url', type=str, default='http://localhost:8002/v1', help='Base URL')

args = parser.parse_args()

task = args.task
model = args.model
noise_ratio = args.noise_ratio
base_url = args.base_url

if model not in MODELS_CONFIG:
    raise ValueError(f"Model {model} not found in MODELS_CONFIG")
if task not in TASK_CONFIG:
    raise ValueError(f"Task {task} not found in TASK_CONFIG")

model_config = MODELS_CONFIG[model]

os.environ['LLM_BASE_URL'] = base_url
if 'OPENAI_API_KEY' in model_config:
    os.environ['OPENAI_API_KEY'] = model_config['OPENAI_API_KEY']

infer_config = {
    'type': model_config["method"],
    'task': task,
    'config': {
        "model": model_config['name'],
        "temperature": 1,
        "max_tokens": 512,
        "logprobs": True
    }
}

eval_config = TASK_CONFIG[task]
question_type = eval_config['question_type']
check_fn = eval_config['check_fn']

# Noisy Path
noisy_dir = f'./data/{task}/noisy'
noisy_labeled_path = f'{noisy_dir}/noisy{noise_ratio}.csv'
# Denoise Path
denoise_dir = f'./data/{task}-{model}/denoise'
os.makedirs(denoise_dir, exist_ok=True)
denoise_path = f'{denoise_dir}/denoise{noise_ratio}.csv'
# labeled data path
labeled_path = f'./data/{task}/labeled.csv'

# Read Noisy Data
if not os.path.exists(noisy_labeled_path):
    raise FileNotFoundError(f"Noisy labeled file not found: {noisy_labeled_path}")

try:
    all_data_df = pd.read_csv(noisy_labeled_path)
except Exception as e:
    raise Exception(f"Error reading noisy labeled file: {e}")

all_data = []

for row1 in all_data_df.iterrows():
    d = row1[1].to_dict()
    d['question_type'] = question_type.lower()
    d['additional_prompt'] = eval_config['additional_prompt']
    all_data.append(d)

## Only for Debug
# num_samples = 10
# all_data = random.Random(0).sample(all_data, num_samples)


"""
Noise Detection
"""

vanilla_predictions = []
reasoning_enhanced_predictions = []

"""
Vanilla inference
"""
inference_data = copy.deepcopy(all_data)
vanilla_eval = Eval(samples=inference_data, **infer_config)
_ = vanilla_eval.eval(
    format_fn=format_question_vanilla, 
    check_fn=eval_config['check_fn'], 
    extract_fn=extract_result
)
vanilla_predictions = vanilla_eval.get_results()


"""
Reasoning-enhanced inference
"""
inference_data = copy.deepcopy(all_data)
reasoning_enhanced_eval = Eval(samples=inference_data, **infer_config)
_ = reasoning_enhanced_eval.eval(
    format_fn=format_reasoning_enhance_question, 
    check_fn=eval_config['check_fn'], 
    extract_fn=extract_result
)
reasoning_enhanced_predictions = reasoning_enhanced_eval.get_results()
reasoning_enhanced_entropy_list = [calculate_entropy(infer['logprobs']) for infer in reasoning_enhanced_predictions]

"""
Data Split
"""
clean_samples = []
noisy_samples = []

data_packed = []
for i in range(len(vanilla_predictions)):
    data_ = all_data[i]
    potential_answer = data_['answer']
    vanilla_pred = vanilla_predictions[i]['PredAnswer']
    reasoning_pred = reasoning_enhanced_predictions[i]['PredAnswer']
    reasoning_entropy = reasoning_enhanced_entropy_list[i]

    data_['vanilla_prediction'] = vanilla_pred
    data_['reasoning_prediction'] = reasoning_pred
    data_['potential_answer'] = potential_answer
    data_['reasoning_entropy'] = reasoning_entropy

    if check_fn(vanilla_pred, potential_answer) and check_fn(reasoning_pred, potential_answer):
        data_['PseudoLabel'] = potential_answer
        data_['clean_flag'] = 1
        clean_samples.append(data_)
    else:
        data_['clean_flag'] = 0
        noisy_samples.append(data_)

"""
Noise Data Re-labeling
Context-enhanced relabeling
"""

nr = NearestReference(k=3)
nr.embed_data(clean_samples, f'tmp/{task}_noisy{noise_ratio}_clean')

def format_context_enhance_question_with_reference(data):
    ref_str = nr.fewshot(data)
    data['reference'] = ref_str
    data['additional_prompt'] = data['additional_prompt']
    data['question_type'] = data['question_type']
    context_user_prompt = format_context_enhance_question(data)
    return context_user_prompt

context_data = copy.deepcopy(noisy_samples)
context_enhance_eval = Eval(samples=context_data, **infer_config)
context_enhance_acc = context_enhance_eval.eval(
    format_fn=format_context_enhance_question_with_reference,
    check_fn=eval_config['check_fn'],
    extract_fn=extract_result
)
context_enhance_res_list = context_enhance_eval.get_results()
context_enhance_entropy_list = [calculate_entropy(infer['logprobs']) for infer in context_enhance_res_list]

"""
Review
"""
data_for_review = []
for i in range(len(context_enhance_res_list)):
    data_ = noisy_samples[i]
    data_['context_prediction'] = context_enhance_res_list[i]['PredAnswer']
    data_for_review.append(data_)

review_data = copy.deepcopy(data_for_review)
review_eval = Eval(samples=review_data, **infer_config)
_ = review_eval.eval(
    format_fn=format_review_question, 
    check_fn=check_answer, 
    extract_fn=extract_result
)

"""
Data Selection
"""
review_res_list = review_eval.get_results()
review_entropy_list = [calculate_entropy(infer['logprobs']) for infer in review_res_list]

data_to_sft = []
for i in range(len(review_res_list)):
    data_ = data_for_review[i]
    data_['context_entropy'] = context_enhance_entropy_list[i]
    data_['reasoning_entropy'] = reasoning_enhanced_entropy_list[i]
    data_['review_entropy'] = review_entropy_list[i]
    data_['PredAnswer'] = review_res_list[i]['PredAnswer']
    data_['PseudoLabel'] = review_res_list[i]['PredAnswer']

    if data_['context_entropy'] < np.percentile(context_enhance_entropy_list, 50):
        data_to_sft.append(data_)

combined_samples = clean_samples + data_to_sft

combined_df = pd.DataFrame(combined_samples)
combined_df.to_csv(denoise_path, index=False)

print(f'=== Noise Data Re-labeling ===')
print(f'Combined Samples: {len(combined_samples)}')
print(f'Save to {denoise_path}')
print(f'========================')

