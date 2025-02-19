from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from typing import List


def ngram_repetition_penalty(
    completions: List[str],
    N: int = 50,
    penalty: float = -0.025,
    **kwargs
) -> List[float]:
    """
    Modified n-gram repetition penalty function:
    
    After tokenizing each string into words, applies a cumulative penalty to each token
    within an n-gram of length N whenever that n-gram appears more than once.
    If a token appears in multiple repeated n-grams, the penalties for that token are summed.
    
    Args:
        completions (List[str]): List of strings to process.
        N (int): Length of n-grams.
        penalty (float): Penalty value to apply for each repeated n-gram token.
        
    Returns:
        List[float]: List of total (cumulative) penalties for each string.
                     0 if no repetitions, negative value of (number of repeated tokens × penalty)
                     if repetitions exist.
    """
    rewards = []  # List to store the final scalar reward

    for content in completions:
        # Tokenize the string into words
        tokens = content.split()
        l = len(tokens)
        # Vector to store penalty for each token (initial value 0)
        r = [0.0] * l
        seen = {}  # Store the first occurrence of each n-gram (or simply its presence)

        # Iterate over each n-gram from index 0 to l-N
        for j in range(l - N + 1):
            # Current n-gram (tuple form)
            ng = tuple(tokens[j:j+N])
            if ng in seen:
                # If the n-gram has already been seen, accumulate penalty for each token in the n-gram
                for t in range(j, j + N):
                    # Check if the token index t is within the range, then accumulate
                    if t < l:
                        r[t] += penalty
            else:
                seen[ng] = j  # 첫 출현 기록

        # The total penalty for each string is the sum of the reward vector
        if (l - N + 1) == 0:
            total_penalty = 0
        else:
            total_penalty = sum(r) / (l - N + 1)
        rewards.append(total_penalty)

    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = completions
    rewards = []
    i = 0
    for content, sol in zip(contents, solution):
        gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
                if reward == 0:
                    reward = -1.0
            except Exception as e:
                print(f"Error math verifying: {e}\nanswer_parsed : {answer_parsed} | gold : {gold_parsed}\n================================================\n")
                reward = -1.0
            # if reward == 0:
            #     print(f"\033[91m{kwargs['problem'][i]}\n{content}\n ->  {answer_parsed} | {gold_parsed}\033[0m\n================================================\n")
            # else:
            #     print(f"\033[92m{kwargs['problem'][i]}\n{content}\033[0m\n================================================\n")
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
        i += 1
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<\|begin_of_thought\|>.*?<\|end_of_thought\|><\|begin_of_solution\|>.*?<\|end_of_solution\|>$" 
    completion_contents = ["<|begin_of_thought|>" + completion for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [0.1 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "repetition_penalty": ngram_repetition_penalty,
}

if __name__ == "__main__":


    print(ngram_repetition_penalty())

    print(format_reward())
