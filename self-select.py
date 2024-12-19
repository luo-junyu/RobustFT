import argparse
import os
import pandas as pd
import copy
from eval import Eval

from common import *
from config import *

SELECT_TEMPLATE = """
You are a critical check expert. Your task is to carefully examine a question and its various predictions to determine if the provided potential answer(ground truth) might be noisy or incorrect.

Question Information:
Question: {question}
{options_str}

Potential Answer (Maybe noisy): {potential_answer}

Please analyze above information and determine if the potential answer is noisy/incorrect.

Your Options:
- Y: the potential answer appears to be noisy/incorrect.
- N: the potential answer seems reliable.

Your response must follow this format:

Answer: [Y/N] (Output the answer directly without any spaces or other punctuation. )
""".strip()


def format_selfselect_question(data):
    return SELECT_TEMPLATE.format(
        question=data['question'],
        options_str=format_option_str(data),
        potential_answer=data['answer'],
    )

def extract_selfselect_result(response):
    answer = extract_result(response)
    return answer

parser = argparse.ArgumentParser(description='Noisy Free Fine-tuning')
parser.add_argument('--task', type=str, default='mmlu', help='Task name')
parser.add_argument('--model', type=str, default='llama3.1', help='Model name')
parser.add_argument('--noise_ratio', type=int, default=50, help='Noise ratio')
parser.add_argument('--base_url', type=str, default='http://localhost:8002/v1', help='Base URL')

args = parser.parse_args()

# Replace hardcoded values with command line arguments
task = args.task
model = args.model
noise_ratio = args.noise_ratio
base_url = args.base_url

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

noisy_dir = f'./data/{task}/noisy'
noisy_labeled_path = f'{noisy_dir}/noisy{noise_ratio}.csv'

selfselect_dir = f'./data/{task}/selfselect'
os.makedirs(selfselect_dir, exist_ok=True)
selfselect_path = f'{selfselect_dir}/selfselect{noise_ratio}.csv'

df = pd.read_csv(noisy_labeled_path, sep=',')
all_data = []

for _, row in df.iterrows():
    data = row.to_dict()
    data['question_type'] = question_type.lower()
    data['additional_prompt'] = eval_config['additional_prompt']
    all_data.append(data)


inference_data = copy.deepcopy(all_data)
selfselect_eval = Eval(samples=inference_data, **infer_config)
_ = selfselect_eval.eval(
    format_fn=format_selfselect_question,
    check_fn=check_fn,
    extract_fn=extract_selfselect_result
)
selfselect_predictions = selfselect_eval.get_results()

clean_data = []

for i in range(len(selfselect_predictions)):
    data = all_data[i]
    selfselect_pred = selfselect_predictions[i]['PredAnswer']
    if selfselect_pred.startswith('N'):
        clean_data.append(data)

clean_data_df = pd.DataFrame(clean_data)
clean_data_df.to_csv(selfselect_path, index=False)
