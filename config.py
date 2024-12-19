from common import *

MODELS_CONFIG = {
    "llama3.1-8b": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "url": "Your VLLM server URL",
        "method": "loop"
    }
}


TASK_CONFIG = {
    "mmlu": {
        "dataset_name": "mmlu",
        "test_path": "./data/mmlu/test.csv",
        "noisy_path": "./data/mmlu/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "The answer must be uppercase letter.",
        "check_fn": check_answer
    },
     "mmlu_pro": {
        "dataset_name": "mmlu_pro",
        "test_path": "./data/mmlu_pro/test.csv",
        "noisy_path": "./data/mmlu_pro/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "The answer must be uppercase letter.",
        "check_fn": check_answer
    },
    "arc": {
        "dataset_name": "arc",
        "test_path": "./data/arc/test.csv",
        "noisy_path": "./data/arc/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "The answer must be uppercase letter.",
        "check_fn": check_answer
    },
    "pubmedqa": {
        "dataset_name": "pubmedqa",
        "test_path": "./data/pubmedqa/test.csv",
        "noisy_path": "./data/pubmedqa/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "The answer must be uppercase letter.",
        "check_fn": check_answer
    },
    "usmle": {
        "dataset_name": "usmle",
        "test_path": "./data/usmle/test.csv",
        "noisy_path": "./data/usmle/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "The answer must be uppercase letter.",
        "check_fn": check_answer
    },
    "fpb": {
        "dataset_name": "fpb",
        "test_path": "./data/fpb/test.csv",
        "noisy_path": "./data/fpb/noisy.csv",
        "question_type": "multi-choice",
        "additional_prompt": "The answer must be uppercase letter.",
        "check_fn": check_answer
    },
    "convfinqa": {
        "dataset_name": "convfinqa",
        "test_path": "./data/convfinqa/test.csv",
        "noisy_path": "./data/convfinqa/noisy.csv",
        "question_type": "math and value extraction",
        "additional_prompt": "The answer should be in digits.",
        "check_fn": check_answer_value
    },
    "drop": {
        "dataset_name": "drop",
        "test_path": "./data/drop/test.csv",
        "noisy_path": "./data/drop/noisy.csv",
        "question_type": "reading comprehension",
        "additional_prompt": "The answer should be a single word or in digits.",
        "check_fn": check_answer_fuzzy
    }
}

