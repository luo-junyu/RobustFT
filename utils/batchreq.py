import os
import openai
openai.api_key = os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else None
import json
import time
from datetime import datetime

def batch_query_openai_chat_model(instances, config, save_dir=None):
    evaluator = BatchRequest(config)
    evaluator.creat_batch_task(instances)
    evaluator.check_until_completed()
    res_list = evaluator.export_batch_result(output_path=save_dir)
    return res_list

def get_logprob_array(probs):
    prob_list = [ins['logprob'] for ins in probs]
    return prob_list


class BatchRequest:
    def __init__(self, config={}) -> None:
        self.client = openai.OpenAI()
        self.config = config
        self.messages_list = []
        self.results_list = []
        self.batch_job = None
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.tmp_file_path = './tmp/batch_msgs_' + timestamp + '.jsonl'
        self.res_file_path = './tmp/batch_res_' + timestamp + '.jsonl'
        os.makedirs('./tmp', exist_ok=True)

    def pack_batch_msgs(self, messages_list):
        tasks = []
        for index, msg in enumerate(messages_list):
            body = {}
            body.update(self.config)
            body['messages'] = msg
            task = {
                "custom_id": f"task-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            tasks.append(task)
        
        with open(self.tmp_file_path, 'w') as file:
            for obj in tasks:
                file.write(json.dumps(obj) + '\n')
        
        return self.tmp_file_path

    def upload_batch_file(self):
        assert os.path.exists(self.tmp_file_path)
        batch_file = self.client.files.create(
            file=open(self.tmp_file_path, "rb"),
            purpose="batch"
        )
        os.remove(self.tmp_file_path)
        return batch_file

    def creat_batch_task(self, messages_list):
        self.messages_list = messages_list
        self.pack_batch_msgs(messages_list)
        batch_file = self.upload_batch_file()

        self.batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f'Batch Task Created: {self.batch_job.id}')

        return self.batch_job
    
    def check_batch_task(self):
        if self.batch_job is None:
            return False
        task = self.client.batches.retrieve(self.batch_job.id)
        status = task.status
        print(f'Batch Task Status: {status}')
        if status == 'completed':
            return True
        return False
    
    def check_until_completed(self):
        while not self.check_batch_task():
            # sleep
            time.sleep(10)
        return True

    def get_batch_result(self):
        task = self.client.batches.retrieve(self.batch_job.id)
        assert task.status == 'completed'
        result_file_id = task.output_file_id
        result = self.client.files.content(result_file_id).content
        return result

    def export_batch_result(self, output_path=None):
        result = self.get_batch_result()
        lines = result.decode().split('\n')

        results_dict = {}
        logprobs_dict = {}
        logprobs_flag = True if 'logprobs' in self.config and self.config['logprobs'] else False

        for line in lines:
            if line:
                res = json.loads(line)
                results_dict[res["custom_id"]] = res['response']['body']['choices'][0]['message']['content']
                if logprobs_flag:
                    logprobs_dict[res["custom_id"]] = get_logprob_array(res['response']['body']['choices'][0]['logprobs']['content'])

        self.results_list = []
        for i, s in enumerate(self.messages_list):
            test_index =f'task-{i}'
            if test_index in results_dict:
                response_text = results_dict[test_index]
            else:
                response_text = ''
            
            if logprobs_flag:
                logprobs_value = logprobs_dict[test_index]
            else:
                logprobs_value = []
            
            self.results_list.append(
                {
                    "response": response_text,
                    "logprobs": logprobs_value 
                } if logprobs_flag else {"response": response_text}
            )

        if output_path is None:
            output_path = self.res_file_path

        with open(output_path, 'w') as file:
            for obj in self.results_list:
                file.write(json.dumps(obj) + '\n')

        return self.results_list