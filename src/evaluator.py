import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from rouge_score import rouge_scorer

ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])


MCQ_DATASETS = ['hellaswag', 'pro_medicine', 'formal_logic', 'csqa', 'hh_rlhf']


def get_instruction_suffix_for_data(data, bae=False, cot=False):
    if data in ['arithmetics']:
        if bae:
            return ' Make sure to state your answer at the end of the response.'
        elif cot:
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
        else:
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 12.34}".'
    elif data in ['gsm8k']:
        if bae:
            return ' Make sure to state your answer at the end of the response.'
        elif cot:
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else:
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif data in MCQ_DATASETS:
        # strict_mcq_suffix = (
        #     " Your final answer MUST be exactly in this format with no extra formatting:\n"
        #     "{final answer: (X)}\n"
        #     "Do NOT use LaTeX, \\boxed, \\text, or any other markup. Just plain text inside curly brackets."
        # )
        strict_mcq_suffix = (
            ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (X)}". Do NOT use LaTeX, \\boxed, \\text, or any other markup. Just plain text inside curly brackets.'
        )
        if bae:
            # return ' Put your final answer in the form (X) at the end of your response.'
            return strict_mcq_suffix
        elif cot:
            # return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
            return strict_mcq_suffix
        else:
            # return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
            return strict_mcq_suffix
    elif data in ['cnn_daily']:
        return ' Make sure to provide your summary after stating "# Summary # ".'


def get_instruction_suffix(args):
    return get_instruction_suffix_for_data(args.data, bae=args.bae, cot=args.cot)


def extract_numeric_final_answer(response):
    try:
        pred = re.findall(r"\{(.*?)\}", response)[-1]
        pred = float(pred.replace("final answer:", "").strip())
        return np.round(pred, 1)
    except:
        return ""


def extract_mcq_final_answer(response):
    try:
        pred = re.findall(r"\{(.*?)\}", response)[-1]
        pred = pred.replace("final answer:", "").strip()
        if len(pred) == 0:
            return ""
        elif len(pred) < 3:
            pred = pred[0]
        else:
            pred = pred[1]
        return f"({pred})"
    except:
        return ""


def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():
        final_answers.append(extract_numeric_final_answer(response))

    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else:
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common)  # if there is a tie, will choose randomly

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():
        final_answers.append(extract_mcq_final_answer(response))

    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else:
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common)  # if there is a tie, will choose randomly
    return final_answers, debate_answer, debate_answer == answer


def evaluate_gen(responses, answer):
    final_answers = []
    debate_answer = None
    best_score = 0
    for _, input_str in responses.items():
        summary = input_str.split("# Summary #")[-1]
        final_answers.append(summary)

        scores = ROUGE.score(answer, summary)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        if rougeL > best_score:
            debate_answer = summary
            best_scores = (rouge1, rouge2, rougeL)

    return final_answers, debate_answer, best_scores


def base_evaluate_arithmetics(responses, answer):
    final_answers = []
    for _, sentence in responses.items():
        parts = sentence.split(" ")

        for part in parts[::-1]:
            try:
                ans = float(part)
                final_answers.append(ans)
                break
            except:
                continue

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common)  # if there is a tie, will choose randomly
    except:
        debate_answer = ""

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def base_evaluate_mcq(responses, answer):
    final_answers = []
    for _, input_str in responses.items():
        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                final_answers.append(f"({solution})")
                break

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common)  # if there is a tie, will choose randomly
    except:
        debate_answer = ""
    return final_answers, debate_answer, debate_answer == answer
