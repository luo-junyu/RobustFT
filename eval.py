import random
import re
import pandas as pd
import sys
sys.path.append('..')
import json
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from common import *
from utils import *
from config import *
import datetime


class Eval:

    def __init__(self, type='loop', samples=[], data_path=None, num_samples=None, config=None, task=''):
        """
        Args:
            type: 'loop' or 'batch'
            samples: a list of samples
            data_path: a path to a csv file
            num_samples: the number of samples to be evaluated
            config: the configuration of the model
        """
        
        if data_path:
            df = pd.read_csv(data_path)
            samples = [row.to_dict() for _, row in df.iterrows()]
        elif not samples:
            raise ValueError("Either samples or data_path must be provided")

        if config is None:
            raise ValueError("config must be provided")
        
        model = config['model'].replace('/', '_')
        task = task
        
        self.config = config
        
        if num_samples:
            samples = random.Random(0).sample(samples, num_samples)
        self.samples = samples

        assert type in ['batch', 'loop']

        self.type = type
        self.results = None
        timestamp = datetime.datetime.now().strftime("%m%d_%H")
        self.output_path = f'./save/{task}_{model}_{timestamp}.json'
        self.gptreq = None
    
    def multiple_inference(self, instances, extract_fn):
        if not self.gptreq:
            self.gptreq = LoopRequest()
        res_list = self.gptreq.batch_req(instances, self.config, save=True, save_dir=self.output_path)

        assert len(res_list) == len(self.samples)

        for i, s in enumerate(self.samples):
            response = res_list[i]['response']
            self.samples[i]["Pred"] = response
            self.samples[i]["PredAnswer"] = extract_fn(response)
            if "logprobs" in res_list[i]:
                self.samples[i]["logprobs"] = res_list[i]["logprobs"]
            # self.samples[i]["PredIndex"] = extract_result_index(response)

    def batch_inference(self, instances, extract_fn):
        res_list = batch_query_openai_chat_model(instances, self.config, save_dir=self.output_path)

        assert len(res_list) == len(self.samples)
        
        for i, s in enumerate(self.samples):
            response = res_list[i]['response']
            self.samples[i]["Pred"] = response
            self.samples[i]["PredAnswer"] = extract_fn(response)
    
    def extract_results(self):
        return self.samples

    def eval(self, format_fn=format_question_vanilla, check_fn=check_answer, extract_fn=extract_result):
        print(f'Formating {len(self.samples)} questions ...')
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            instances = list(tqdm(executor.map(lambda x: [{"role": "user", "content": format_fn(x)}], self.samples), total=len(self.samples)))

        ## keep
        # for row in tqdm(self.samples):
        #     instances.append([{"role": "user", "content": format_fn(row)}])

        print(f'Begin Inference ...')

        if self.type == 'loop':
            self.multiple_inference(instances, extract_fn)
        else:
            self.batch_inference(instances, extract_fn)
        
        cors = []
        for i, s in enumerate(self.samples):
            score = 1.0 if check_fn(s['Pred'], s["answer"]) else 0.0
            cors.append(score)

        acc = np.mean(cors)
        return acc

    def get_results(self):
        return self.samples

    
if __name__ == "__main__":
    task_model_list = [
        ['mmlu', 'llama3.1-8b']
    ]

    for task, model in task_model_list:

        if not task in TASK_CONFIG or not model in MODELS_CONFIG:
            print(f'{task} or {model} not found')
            continue

        model_config = MODELS_CONFIG[model]
        task_config = TASK_CONFIG[task]
        
        os.environ['LLM_BASE_URL'] = model_config["url"]
        if 'OPENAI_API_KEY' in model_config:
            os.environ['OPENAI_API_KEY'] = model_config['OPENAI_API_KEY']
            
        # Prepare test samples
        test_df = pd.read_csv(task_config['test_path'])
        
        ## For Debugging
        # num_samples = 10
        # test_df = test_df.sample(num_samples)
        
        test_samples = []
        for _, row in test_df.iterrows():
            d = row.to_dict()
            d['question_type'] = task_config['question_type']
            d['additional_prompt'] = task_config['additional_prompt']
            test_samples.append(d)
        
        # Inference and evaluate configuration 
        infer_config = {
            'type': model_config["method"],
            'task': task,
            'config': {
                "model": model_config['name'],
                "temperature": 0.5,
                "max_tokens": 1000,
                "logprobs": True
            },
            'samples': test_samples
        }
        
        eval_config =  {
            'format_fn': format_question_vanilla, 
            'check_fn': task_config['check_fn'], 
            'extract_fn': extract_result
        }

        eval = Eval(**infer_config)
        acc = eval.eval(**eval_config)
        
        print(f'Accuracy: {acc}')