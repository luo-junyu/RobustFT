import re
import json

"""
Vanilla Inference
"""

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following {question_type} question directly. Your answer must be on a new line starting with exactly "Answer: ". Put your answer immediately after "Answer: " without any spaces or other punctuation. {additional_prompt}

Question: 
{question}

{options_str}
""".strip()

# ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-Z])"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\s\n]+)"

def format_option_str(row):
    if not 'options' in row:
        return ''
    options = row['options']
    if type(options) == str:
        options = json.loads(options)
    options = [f"{chr(65+i)}. {option}" for i, option in enumerate(options)]
    options_str = "\n".join(options)
    return f'Options:\n{options_str}\n'

def format_question_vanilla(row):
    question = row['question']
    options_str = format_option_str(row)
    question_type = row['question_type']
    additional_prompt = row['additional_prompt']
    return QUERY_TEMPLATE_MULTICHOICE.format(question=question, options_str=options_str, question_type=question_type, additional_prompt=additional_prompt)

def extract_result(res):
    match = re.search(ANSWER_PATTERN, res)
    extracted_answer = match.group(1) if match else res[0].upper()
    return extracted_answer

"""
Check Answer
"""

def check_answer(res, gt):
    pred = extract_result(res)
    if len(pred) < len(gt):
        pred = pred[:len(gt)]
    elif len(pred) > len(gt):
        gt = gt[:len(pred)]
    return pred == gt

def normoalize_num(num):
    def eval_num(num):
        num = num.replace('%','/100').replace(',','')
        try:
            num = eval(num)
        except Exception as e:
            num = float('inf')
            pass
        return num
    VALUE_PATTERRN = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?[%]*"
    val_reg = re.compile(VALUE_PATTERRN)
    return [eval_num(num) for num in val_reg.findall(num)]


def check_value_equal(res_arr, gt_arr):
    import math
    for gt_num in gt_arr:
        for pred_num in res_arr:
            if math.isclose(pred_num, gt_num, rel_tol=1e-2):
                return True
    return False

def check_answer_value(res, gt):
    pred = normoalize_num(extract_result(res))
    gt = normoalize_num(gt)
    return check_value_equal(pred, gt)

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    import string
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1

def check_answer_fuzzy(res: str, gt: list):
    pred = extract_result(res)
    match_list = [fuzzy_match(pred, gt_item) for gt_item in gt]
    return True in match_list

"""
Context-enhanced Inference
"""

CONTEXT_ENHANCE_TEMPLATE = """
Here are some examples of questions and their corresponding answer.
{reference}

Answer the following {question_type} question. Your answer must be on a new line starting with exactly "Answer: ". Put your answer immediately after "Answer: " without any spaces or other punctuation. {additional_prompt}

Question: 
{question}

{options_str}
""".strip()

def format_question_and_answer(row):
    question_str = f'Question: {row["question"]}\n'
    options_str = format_option_str(row)
    answer_str = f'Answer: {row["answer"]}'
    return f"{question_str}{options_str}{answer_str}"

def get_the_shortest_str_inlist(str_list):
    return min(str_list, key=len)

def format_context_enhance_question(row):
    reference_str = row['reference']
    question = row['question']
    if type(question) == list:
        question = get_the_shortest_str_inlist(question)
    question_type = row['question_type']
    additional_prompt = row['additional_prompt']
    options_str = format_option_str(row)
    
    return CONTEXT_ENHANCE_TEMPLATE.format(reference=reference_str, question=question, question_type=question_type, additional_prompt=additional_prompt, options_str=options_str)

"""
Reasoning-enhanced Inference
"""

REASONING_ENHANCE_TEMPLATE = """
Answer the following {question_type} question. 

IMPORTANT: 
1. You can include your step-by-step reasoning and reflection, and provide the final answer directly. 
2. Your final answer must be on a new line starting with exactly "Answer: ". Put your answer immediately after "Answer: " without any spaces or other punctuation. {additional_prompt}
3. Your reasoning process should be clear and concise.

Question: 
{question}

{options_str}
""".strip()

def format_reasoning_enhance_question(row):
    question = row['question']
    question_type = row['question_type']
    options_str = format_option_str(row)
    return REASONING_ENHANCE_TEMPLATE.format(
        question=question, 
        question_type=question_type, 
        options_str=options_str,
        additional_prompt=row.get('additional_prompt', '')
    )


"""
Check Agent
"""

CHECK_TEMPLATE = """
You are a critical check expert. Your task is to carefully examine a question and its various predictions to determine if the provided potential answer(ground truth) might be noisy or incorrect.

Question Information:
Question: {question}
{options_str}

1. Potential Answer (Maybe noisy): {potential_answer}

2. Basic Prediction: {vanilla_pred}
Reasoning: {vanilla_reasoning}

3. Reasoning-Enhanced Prediction: {reasoning_pred}
Reasoning: {reasoning_reasoning}

Please analyze this carefully and compare all predictions with the potential answer. The noisy answer is likely to be inconsistent with the majority of predictions.

Your Options:
- Y: the potential answer appears to be noisy/incorrect.
- N: the potential answer seems reliable.

Your response must follow this format:

Answer: [Y/N] (Output the answer directly without any spaces or other punctuation. )
""".strip()

def format_check_question(data):
    return CHECK_TEMPLATE.format(
        question=data['question'],
        options_str=format_option_str(data),
        potential_answer=data['potential_answer'],
        vanilla_pred=data['vanilla_prediction'],
        vanilla_reasoning=data.get('vanilla_reasoning', 'No reasoning provided'),
        reasoning_pred=data['reasoning_prediction'],
        reasoning_reasoning=data.get('reasoning_reasoning', 'No reasoning provided')
    )

def extract_check_result(response):
    answer = extract_result(response)
    return answer


"""
Review Agent

Input with reasoning-enhanced prediction and context-enhanced relabeling.
Output with the final output for SFT.
"""

REVIEW_TEMPLATE = """
You are a review expert to predict the final answer of {question_type} questions.

Question Information:
Question: {question}
{options_str}

1. Reasoning-Enhanced Prediction:
{reasoning_pred}

2. Context-Enhanced Prediction:
{context_pred}

Please directly give me the final answer. The answer must be starting with exactly "Answer: ". Put your answer immediately after "Answer: " without any spaces or other punctuation. {additional_prompt}
""".strip()


def format_review_question(data):
    return REVIEW_TEMPLATE.format(
        question=data['question'],
        options_str=format_option_str(data),
        reasoning_pred=data['reasoning_prediction'],
        reasoning_reasoning=data.get('reasoning_reasoning', 'No reasoning provided'),
        context_pred=data['context_prediction'],
        additional_prompt=data.get('additional_prompt', ''),
        question_type=data['question_type']
    )
