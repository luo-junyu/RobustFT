import pandas as pd
import json
from common import *
from config import *
import os
import ast


def format_question_alpaca(row, format_fn=format_question_vanilla):
    row = row.to_dict()
    row['question_type'] = eval_config['question_type']
    row['additional_prompt'] = eval_config['additional_prompt']
    input_text = format_fn(row)
    output_test = pack_answer(row)
    return {
        "instruction": input_text,
        "input": '',
        "output": output_test
    }

def format_qa_gpt(row, format_fn=format_question_vanilla):
    return {
        'messages': [
            {"role": "user", "content": format_fn(row)},
            {"role": "assistant", "content": pack_answer(row)}
        ]
    }

def format_gpt_eval(row, format_fn=format_question_vanilla):
    return {
        'question': format_fn(row),
        'answer': pack_answer(row)
    }

def parse_string(s):
    s = s.replace("array(", "").replace(", dtype=object)", "")
    return ast.literal_eval(s)

def pack_answer(row):
    if 'PseudoLabel' in row:
        pl = row['PseudoLabel']
    else:
        pl = row['answer']

    if "' '" in pl or '" "' in pl:
        pl = pl.strip('[]').split("' '")
        pl = [item.strip("' ") for item in pl]
    
    if isinstance(pl, str) and pl.startswith('['):
        pl = parse_string(pl)

    if isinstance(pl, list):
        pl = get_the_shortest_str_inlist(pl)

    pl = pl.replace('"', '').replace("'", '').replace('[]', '')
    pl = pl.rstrip('. ').strip()
    return f'Answer: {pl}'

if __name__ == '__main__':
    # task = 'drop'
    task_list = ['mmlu']
    model='llama3.1'
    output_format = 'alpaca'

    for task in task_list:
        for noisy_ratio in [30, 50, 70]:
            datatype = f'denoise/denoise{noisy_ratio}'
            input_file = f'./data/{task}/{datatype}.csv'

            if not os.path.exists(input_file):
                continue

            print(f'Processing {input_file}')
            output_file = f'./data/{task}-{model}/{datatype}_{output_format}.json'
            eval_config = TASK_CONFIG[task]
            format_fn=format_question_vanilla
            check_fn=eval_config['check_fn']
            extract_fn=extract_result

            df = pd.read_csv(input_file)

            examples = [format_question_alpaca(row, format_fn) for _, row in df.iterrows()]
            with open(output_file, 'w') as f:
                json.dump(examples, f, indent=2)
          
            
            print(f'Finished formatting {len(examples)} examples to {output_file}')