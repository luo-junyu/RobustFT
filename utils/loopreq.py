from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import time
import json
import os 

def get_tmp_file_path():
    created_time = time.time()
    created_time = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    res_file_path = './tmp/batch_res_' + timestamp + '.jsonl'
    return res_file_path

def get_logprob_array(probs):
    prob_list = [ins.logprob for ins in probs]
    return prob_list

class LoopRequest:
    def __init__(self):
        openai_api_key = os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else "EMPTY"
        openai_api_base = os.environ['LLM_BASE_URL'] if 'LLM_BASE_URL' in os.environ else "https://api.openai.com/v1"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.results_list = []


    def single_req(self, msg, config, logprobs=False):
        chat_completion = self.client.chat.completions.create(
            messages=msg,
            **config
        )
        if 'logprobs' in config and config['logprobs']:
            return chat_completion.choices[0].message.content, get_logprob_array(chat_completion.choices[0].logprobs.content)
        else:
            return chat_completion.choices[0].message.content
        
    
    def batch_req(self, messages_list, config, save=False, save_dir=''):
        res_list = []
        logprobs_flag = True if 'logprobs' in config and config['logprobs'] else False

        for msg in tqdm(messages_list):
            res = self.single_req(msg, config)
            res_list.append(
                {   
                    "response": res[0],
                    "logprobs": res[1]
                } if logprobs_flag else {"response": res}
            )
        self.results_list = res_list

        if save:
            output_path = save_dir if save_dir else get_tmp_file_path()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as file:
                for obj in self.results_list:
                    file.write(json.dumps(obj) + '\n')

        return res_list


if __name__=='__main__':
    os.environ['LLM_BASE_URL'] = "http://localhost:8000/v1"
    openaireq = LoopRequest()

    res = openaireq.batch_req(
        messages_list=[
            [{ "role": "user", "content": "Who won the world series in 2020?" }],
            [{ "role": "user", "content": "Who won the world series in 2020?" }],
        ],
        config= {
            "model": 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            "temperature": 1,
            "logprobs": True,
            "max_tokens": 1000,
        }
    )

    print(res)
