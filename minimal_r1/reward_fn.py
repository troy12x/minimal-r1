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
    pattern = (
        r"^\s*"
        r"<think>((?:(?!<think>|</think>)[\s\S])*)</think>"
        r"\s*"
        r"<answer>((?:(?!<answer>|</answer>)[\s\S])*)</answer>"
        r"\s*$"
    )
    completion_contents = ["<think>" + completion for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [0.1 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "repetition_penalty": ngram_repetition_penalty,
}

if __name__ == "__main__":

    completions_0 = """
<think>

... some text ...
</think>

<answer>
maybe

Thus, the final answer is \(\boxed{0}\).

</answer>
"""
    completions_4 = """<think>
The word "strawberry" contains one 'r'.
</think>
<answer>
1
</answer>"""

    completions_1 = """<think>....haha...</think><answer>maybe ... \(\boxed{0}\)</answer>"""
    completions_2 = """<think>....haha maybe</think><answer>maybe ..."""
    completions_3 = """<think>....haha haha maybe. sin(60.."""


    rep_penalty_ex1 = """
tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest tallest ta"""

    completions_5 = """  To solve this problem, we can use the Angle Bisector Theorem and the formula for the length of the angle bisector. The Angle Bisector Theorem states that the angle bisector of a triangle divides the opposite side into segments that are proportional to the adjacent sides. The formula for the length of the angle bisector $CD$ is given by:

\[ CD = \frac{2 \cdot BC \cdot AC \cdot \cos \left( \frac{C}{2} \right)}{BC + AC} \]

Given that $\cos \left( \frac{C}{2} \right) = \frac{1}{3}$ and $CD = 6$, we can substitute these values into the formula:

\[ 6 = \frac{2 \cdot BC \cdot AC \cdot \frac{1}{3}}{BC + AC} \]

Simplifying this equation, we get:

\[ 6 = \frac{2 \cdot BC \cdot AC}{3 \cdot (BC + AC)} \]
\[ 18 \cdot (BC + AC) = 2 \cdot BC \cdot AC \]
\[ 9 \cdot (BC + AC) = BC \cdot AC \]

We need to find the value of $\frac{1}{BC} + \frac{1}{AC}$. Notice that:

\[ \frac{1}{BC} + \frac{1}{AC} = \frac{BC + AC}{BC \cdot AC} \]

From the equation $9 \cdot (BC + AC) = BC \cdot AC$, we can substitute $BC \cdot AC$ with $9 \cdot (BC + AC)$:

\[ \frac{1}{BC} + \frac{1}{AC} = \frac{BC + AC}{9 \cdot (BC + AC)} = \frac{1}{9} \]

Thus, the value of $\frac{1}{BC} + \frac{1}{AC}$ is $\frac{1}{9}$. The final answer is:

\[
\boxed{\frac{1}{9}}
\]</think>
<answer> $\boxed{\frac{1}{9}}$ </answer>"""

    completions_6 = """ The given functional equation is:
\[ f(x + y) = f(x) f(y) \]
We need to find all possible values of \( f(0) \).

Let's substitute \( x = 0 \) and \( y = 0 \) into the equation:
\[ f(0 + 0) = f(0) f(0) \]
This simplifies to:
\[ f(0) = f(0)^2 \]

Let \( f(0) = a \). Then we have:
\[ a = a^2 \]
This is a quadratic equation. Rearranging it, we get:
\[ a^2 - a = 0 \]
Factoring out \( a \), we get:
\[ a(a - 1) = 0 \]
So the solutions are:
\[ a = 0 \quad \text{or} \quad a = 1 \]

Thus, \( f(0) \) can be either \( 0 \) or \( 1 \).

To verify, we need to check if these values are consistent with the original functional equation.

1. If \( f(0) = 0 \):
   Substituting \( x = 0 \) into the original equation, we get:
   \[ f(0 + y) = f(0) f(y) \]
   This simplifies to:
   \[ f(y) = 0 \]
   This means \( f(x) = 0 \) for all \( x \), which is a valid solution.

2. If \( f(0) = 1 \):
   Substituting \( x = 0 \) into the original equation, we get:
   \[ f(0 + y) = f(0) f(y) \]
   This simplifies to:
   \[ f(y) = 1 \cdot f(y) \]
   Which is true for all \( f(y) \). This means \( f(x) = 1 \) for all \( x \), which is also a valid solution.

So the possible values of \( f(0) \) are \( 0 \) and \( 1 \).

The final answer is:
\[
\boxed{0, 1}
\]</think>
<answer> 0, 1 </answer>"""

    completions_7 = """Let's start by defining the properties of the sequences we are interested in. We need sequences of length 14 where every run of consecutive $A$'s has even length, and every run of consecutive $B$'s has odd length.

We can use dynamic programming to solve this problem. Let's define two arrays:
- `dp_A[n]` to store the number of valid sequences of length `n` that end with an $A$.
- `dp_B[n]` to store the number of valid sequences of length `n` that end with a $B$.

The transitions between states can be described as follows:
- To form a sequence of length `n` ending with an $A$, we must append an even number of $A$'s to a sequence of length `n - 2k` ending with a $B$. So, `dp_A[n] = dp_B[n-2] + dp_B[n-4] + dp_B[n-6] + ...`.
- To form a sequence of length `n` ending with a $B$, we must append an odd number of $B$'s to a sequence of length `n - (2k+1)` ending with an $A$. So, `dp_B[n] = dp_A[n-1] + dp_A[n-3] + dp_A[n-5] + ...`.

The base cases are:
- `dp_A[0] = 0` (since we cannot form a sequence of length 0 ending with an $A$).
- `dp_B[0] = 1` (the empty sequence, which ends with an empty string, is considered valid).

We need to find the total number of valid sequences of length 14, which is `dp_A[14] + dp_B[14]`.

Let's implement this in Python:</think>
<answer>Let's start by defining the properties of the sequences we are interested in. We need sequences of length 14 where every run of consecutive $A$'s has even length, and every run of consecutive $B$'s has odd length.

We can use dynamic programming to solve this problem. Let's define two arrays:
- `dp_A[n]` to store the number of valid sequences of length `n` that end with an $A$.
- `dp_B[n]` to store the number of valid sequences of length `n` that end with a $B$.

The transitions between states can be described as follows:
- To form a sequence of length `n` ending with an $A$, we must append an even number of $A$'s to a sequence of length `n - 2k` ending with a $B$. So, `dp_A[n] = dp_B[n-2] + dp_B[n-4] + dp_B[n-6] + ...`.
- To form a sequence of length `n` ending with a $B$, we must append an odd number of $B$'s to a sequence of length `n - (2k+1)` ending with an $A$. So, `dp_B[n] = dp_A[n-1] + dp_A[n-3] + dp_A[n-5] + ...`.

The base cases are:
- `dp_A[0] = 0` (since we cannot form a sequence of length 0 ending with an $A$).
- `dp_B[0] = 1` (the empty sequence, which ends with an empty string, is considered valid).

We need to find the total number of valid sequences of length 14, which is `dp_A[14] + dp_B[14]`.

Let's implement this in Python:

```python
# Define the length of the sequence
length = 14

# Initialize the dp arrays
dp_A = [0] * (length + 1)
dp_B = [0] * (length + 1)

# Base cases
dp_B[0] = 1

# Fill the dp arrays
for n in range(1, length + 1):
    for k in range(1, n//2 + 1):
        dp_A[n] += dp_B[n - 2 * k]
    for k in range(0, n//2):
        dp_B[n] += dp_A[n - (2 * k + 1)]

# The total number of valid sequences of length 14
total_sequences = dp_A[length] + dp_B[length]
print(total_sequences)
```
```output
172
```
The total number of valid sequences of length 14, where every run of consecutive $A$'s has even length and every run of consecutive $B$'s has odd length, is \(\boxed{172}\).

"""

    completions_8 = """Let's define the number of cheese danishes as \( c \) and the number of cherry danishes as \( k \). According to the problem, we have the following inequalities:
\[ \frac{2}{3}c + 3 \leq k \leq 2c \]
We need to find the smallest possible value for the total number of danishes, which is \( c + k \).

We will start by iterating over possible values of \( c \) and checking the corresponding values of \( k \) to find the smallest possible total number of danishes. We'll start with \( c = 1 \) and increment \( c \) until we find a valid solution.

Let's write the code to find the solution.
</think>
<answer>Let's start by iterating over possible values of \( c \) and checking the corresponding values of \( k \) to find the smallest possible total number of danishes.

We'll start with \( c = 1 \) and increment \( c \) until we find a valid solution.
```python
# Initialize the minimum total number of danishes to a large number
min_total = float('inf')
min_c = 0
min_k = 0

# Iterate over possible values of c
for c in range(1, 100):  # We can start with a reasonable range, say 1 to 100
    # Calculate the possible range for k
    k_min = (2/3) * c + 3
    k_max = 2 * c
    
    # Check if there is an integer k that satisfies the inequalities
    for k in range(int(k_min) + 1, int(k_max) + 1):  # Start from the ceiling of k_min to the floor of k_max
        if k_min <= k <= k_max:
            total = c + k
            if total < min_total:
                min_total = total
                min_c = c
                min_k = k

print(((min_total, min_c, min_k)))
```
```output
(11, 3, 8)
```
The smallest possible value for the total number of danishes is \( 11 \). This occurs when the number of cheese danishes is \( 3 \) and the number of cherry danishes is \( 8 \).

So, the final answer is:
\[
\boxed{11}
\]
"""

    print(ngram_repetition_penalty([rep_penalty_ex1, completions_3, completions_5, completions_6, completions_7, completions_8]))

    print(format_reward([completions_0, completions_1, completions_2, completions_3, completions_4]))