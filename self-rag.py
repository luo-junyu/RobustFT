import pandas as pd
import os
from common import *
from eval import Eval
from rag import NearestReference
from config import *
import argparse
import copy

class SelfRAGEval(Eval):
    def __init__(self, k=3, embedding_cache_dir='tmp', **kwargs):
        """
        Args:
            k: number of reference examples to retrieve
            embedding_cache_dir: directory to cache embeddings
            **kwargs: arguments passed to parent Eval class
        """
        super().__init__(**kwargs)
        self.k = k
        self.task = kwargs['task']
        self.cache_dir = embedding_cache_dir
        self.reference_retriever = None
        
    def prepare_references(self, reference_samples):
        """Initialize the reference retriever with examples"""
        self.reference_retriever = NearestReference(k=self.k)
        cache_path = os.path.join(self.cache_dir, f'{self.task}_references')
        self.reference_retriever.embed_data(reference_samples, cache_path)

    def format_with_references(self, sample):
        """Format prompt with retrieved reference examples"""
        if not self.reference_retriever:
            raise ValueError("Reference retriever not initialized. Call prepare_references first.")
            
        references = self.reference_retriever.fewshot(sample)
        sample = copy.deepcopy(sample)
        sample['reference'] = references
        return format_context_enhance_question(sample)

    def eval_with_references(self, reference_samples, format_fn=None, **kwargs):
        """
        Evaluate model performance using retrieved reference examples
        
        Args:
            reference_samples: list of examples to use as references
            format_fn: optional custom format function
            **kwargs: additional arguments passed to eval()
        """
        # Prepare reference retriever
        self.prepare_references(reference_samples)
        
        # Use custom formatter that adds references
        if format_fn is None:
            format_fn = self.format_with_references
            
        # Run evaluation
        return self.eval(format_fn=format_fn, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mmlu')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--noisy_ratio', type=str, default='')
    args = parser.parse_args()

    # Load configs
    model_config = MODELS_CONFIG[args.model]
    task_config = TASK_CONFIG[args.task]
    
    # Set up environment
    os.environ['LLM_BASE_URL'] = model_config['url']
    if 'OPENAI_API_KEY' in model_config:
        os.environ['OPENAI_API_KEY'] = model_config['OPENAI_API_KEY']

    # Load test data
    test_df = pd.read_csv(task_config['test_path'])
    if args.num_samples:
        test_df = test_df.sample(args.num_samples, random_state=42)

    # reference df
    ref_path = f'./data/{args.task}/noisy/noisy{args.noisy_ratio}.csv'
    ref_df = pd.read_csv(ref_path)
    
    test_samples = []
    reference_samples = []

    for _, row in test_df.iterrows():
        d = row.to_dict()
        d['question_type'] = task_config['question_type']
        d['additional_prompt'] = task_config['additional_prompt']
        test_samples.append(d)

    for _, row in ref_df.iterrows():
        d = row.to_dict()
        d['question_type'] = task_config['question_type']
        d['additional_prompt'] = task_config['additional_prompt']
        reference_samples.append(d)

    # Configure evaluation
    eval_config = {
        'type': model_config['method'],
        'task': args.task,
        'config': {
            'model': model_config['name'],
            'temperature': 0.5,
            'max_tokens': 1000,
            'logprobs': True
        },
        'samples': test_samples
    }

    # Run evaluation with references
    evaluator = SelfRAGEval(k=args.k, **eval_config)
    acc = evaluator.eval_with_references(
        reference_samples=reference_samples,
        check_fn=task_config['check_fn'],
        extract_fn=extract_result
    )
    
    print(f'Accuracy with {args.k} references: {acc:.4f}')
    
if __name__ == '__main__':
    main()