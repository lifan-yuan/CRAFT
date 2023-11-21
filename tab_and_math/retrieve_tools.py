import pandas as pd
import numpy as np
import datasets
import openai
from utils import *
from collections import Counter
# openai.api_key = "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"

retrieval_template_tab = """Given a table and a corresponding query, please summary the task goal and briefly describe the row and column of the table.
Next, infer generic table processing tool functions that can achieve the task goal.
Finally, infer the docstring of the tool functions.

Consider following principles:
1. Generic tool function names should be less than eight words in length. Consider utilizing the most frequently used words in function names listed below.
2. The docstring should summaize the task goal and table format. Be general and abstract, not specific to the query. Consider utilizing the most frequently used words in function docstrings listed below.
3. End your answer with the format 'The useful functions are: [...]' and 'The final answer is: ...', where '[...]' is a list of useful functions and '...' is the returned answer.
4. The most frequently used words in function names: ['count', 'difference', 'items', 'total', 'value', 'calculate', 'frequency', 'stem', 'leaf', 'groups', 'table', 'two', 'get', 'item', 'cost', 'specific', 'entities', 'column', 'threshold', 'find', 'group', 'unit', 'probability']
5. The most frequently used words in function docstrings: ['two', 'number', 'item', 'column', 'items', 'specific', 'frequency', 'values', 'name', 'total', 'difference', 'value', 'groups', 'specified', 'table', 'given', 'row', 'stemandleaf', 'based', 'plot', 'entities', 'target', 'names']

*Table*
Name: Orange candies per bag
Unit: bags
Content:
Stem | Leaf 
2 | 2, 3, 9
3 | 
4 | 
5 | 0, 6, 7, 9
6 | 0
7 | 1, 3, 9
8 | 5
*Query*
A candy dispenser put various numbers of orange candies into bags. How many bags had at least 32 orange candies?
Let's think step by step:
To solve the problem, we should count the number of bags with at least a certain threshold of candies based on a stem-and-leaf plot. The code should combine stem and leaf values to calculate the total number of candies in each bag, filter the bags with candies greater than or equal to the threshold value, and count the number of such bags. 
Considering the naming rules of tool functions, the relevant and useful functions could be named as 'count_groups_above_threshold_in_stem_leaf' or 'count_items_above_threshold_based_on_numeric_combination'.
Finally, we can infer that the docstring of the tool function could be 'Given a threshold value and a pandas DataFrame representing a stem-and-leaf plot of groups, count the number of groups that have values greater than or equal to the threshold.'
The useful functions are: ['count_groups_above_threshold_in_stem_leaf', 'count_items_above_threshold_based_on_numeric_combination'].
The final answer is: Given a threshold value and a pandas DataFrame representing a stem-and-leaf plot of groups, count the number of groups that have values greater than or equal to the threshold.


*Table*
pasta with meat sauce | $6.49
pasta with mushrooms | $9.05
spaghetti and meatballs | $7.43
mushroom pizza | $9.28
*Query*
How much money does Jayla need to buy 5 orders of pasta with meat sauce and 3 orders of pasta with mushrooms?
Let's think step by step:
To solve the problem, we should calculate the total cost of items based on a table of item prices per unit and a dictionary of item quantities. The code should calculate the total cost by using the formula total_cost = unit_price * item_quantity for each item, iterating through item names, filtering the table for each item, and calculating the cost based on quantities.
Considering the naming rules of tool functions, the relevant and useful functions could be named as 'calculate_total_cost_from_unit_prices_and_quantities' or 'calculate_total_quantity_from_items_and_coefficients'.
Finally, we can infer that the docstring of the tool function could be 'Calculate the total cost of the items based on a pandas DataFrame representing a table of item prices and a dictionary of item quantities.'
The useful functions are: ['calculate_total_cost_from_unit_prices_and_quantities', 'calculate_total_quantity_from_items_and_coefficients'].
The final answer is: Calculate the total cost of the items based on a pandas DataFrame representing a table of item prices and a dictionary of item quantities.

*Table*
{}
*Query*
{}
Let's think step by step:
"""


retrieval_template_math = """Given a query, please infer the core mathematical skill for the solution.
Next, infer generic mathematical tool functions that can perform the core skill.
Finally, infer the docstring of the tool functions.

Consider the following principles:
1. Generic tool function names should be less than eight mathematic terms in length. Consider utilizing the most frequently used words in function names listed below.
2. The docstring should summarize the task goal. Be general and abstract, not specific to the query. Consider utilizing the most frequently used words in function docstrings listed below.
3. End your answer with the format 'The useful functions are: [...]' and 'The final answer is: ...', where '[...]' is a list of useful functions and '...' is the returned answer.
4. The most frequently used words in function names: ['find', 'calculate', 'sum', 'value', 'expression', 'difference', 'number', 'items', 'total', 'time', 'target', 'inverse', 'generic', 'constant', 'max', 'squares', 'proportional', 'product', 'consecutive', 'evaluate', 'x', 'term', 'factor', 'largest']
5. The most frequently used words in function docstrings: ['number', 'given', 'calculate', 'based', 'two', 'total', 'find', 'value', 'sum', 'time', 'target', 'items', 'certain', 'numbers', 'amount', 'cost', 'first', 'distance']

Query: Let \\[f(x) =\n\\begin{cases}\n3x^2 + 2&\\text{if } x\\le 3, \\\\\nax - 1 &\\text{if } x>3.\n\\end{cases}\n\\]Find $a$ if the graph of $y=f(x)$ is continuous (which means the graph can be drawn without lifting your pencil from the paper).
Let's think step by step:
To solve the problem, we should ensure the continuity of the function by equating the two function expressions at the boundary (x=3). The code should substitute x with 3 in both expressions, equate them, and solve for the unknown variable 'a'.
Considering the naming rules of tool functions, the relevant and useful functions could be named as 'solve_continuous_piecewise_function' or 'find_constant_for_continuity'.
Finally, we can infer that the docstring of the tool function could be 'Find the constant that makes a piecewise function continuous at a given point.'
The useful functions are: ['solve_continuous_piecewise_function', 'find_constant_for_continuity'].
The final answer is: 'Find the constant that makes a piecewise function continuous at a given point.

Query: If $f(3)=1$ and $f(2x)=2f(x)$ for all $x$, find $f^{-1}(64)$
Let's think step by step:
To solve the problem, we should first understand the relationship between the given function and its inverse. Then, we need to use the provided information about the function and its properties to deduce the value of the inverse function at a given input. The code should analyze the properties to establish a connection (which is a linear relationship here) between the function and its inverse, and subsequently evaluate the result for the specific input.
Considering the naming rules of tool functions, the relevant and useful functions could be named as 'find_inverse_value_linear' or 'find_inverse_value_based_on_properties'.
Finally, we can infer that the docstring of the tool function could be 'Find the value of the inverse function based on a linear combination rule for the original function.'
The useful functions are: ['find_inverse_value_linear', 'find_inverse_value_based_on_properties'].
The final answer is: 'Find the value of the inverse function based on a linear combination rule for the original function.

"""



def planning(query, table=None):
    if table is None:
        prompt = retrieval_template_tab.format(table, query)
    else:
        prompt = retrieval_template_math + "\n".join([f"Query: {query}", "Let's think step by step:"]) + "\n"
    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=200,
                    )["choices"][0]["message"]["content"].strip()

    plans = [query, response.split("The final answer is: ")[1].strip()]

    try:
        expected_tools = eval(response.split("\n")[-2].split("The useful functions are: ")[1].strip("."))
    except:
        expected_tools = eval(response.split("\n")[-2].split("The useful function is: ")[1].strip("."))

    return plans, expected_tools


def match_plan_from_single_perspective(plan_embeddings, tool_embeddings, k=3): # k: number of tools to retrieve for each sub-task from each perspective
    tool_list = []
    for plan_embedding in plan_embeddings:
        # compute cos sim between plan and query
        plan_embedding = plan_embedding.unsqueeze(0)
        sim = torch.nn.functional.cosine_similarity(plan_embedding.unsqueeze(1), tool_embeddings.unsqueeze(0), dim=2)
        topk = torch.topk(sim, k=k, dim=1).indices.squeeze(0).tolist()
        tool_list.append(topk)
    return tool_list



def retrieve_tool(example, vector_library, model, tokenizer, k=3): # k: number of tools to retrieve for each sub-task
    print("Question:", example['question'], "\n")
    # decompose the query into sub-tasks
    if "table" not in example.keys():
        plans, expected_tools = planning(example['question'])
    else:
        plans, expected_tools = planning(example['question'], example['table'])
    plan_embeddings = compute_simcse(model, tokenizer, plans)
    expected_tool_embeddings = compute_simcse(model, tokenizer, expected_tools)

    # match plan with tools from different perspectives
    tool_by_explanation = match_plan_from_single_perspective(plan_embeddings[1:], vector_library["explanation_embedding"], k=10)
    tool_by_name = match_plan_from_single_perspective(expected_tool_embeddings, vector_library["name_embedding"], k=5)
    tool_by_query = match_plan_from_single_perspective(plan_embeddings[0].unsqueeze(0), vector_library["query_embedding"], k=10)
    tool_list = []
    print()

    counter = Counter([ 
                        *[item for sublist in tool_by_explanation for item in sublist], # k_1*len(plans)
                        *[item for sublist in tool_by_name for item in sublist], # k_1*len(plans)
                        *[item for sublist in tool_by_query for item in sublist], # k_1*1
                    ])
    top_k = counter.most_common(k) # k_2
    tool_list.extend([tool for (tool, count) in top_k if count >= 2]) # must at least have 2 votes

    tool_list = list(set(tool_list))

    return {"expected_tools": expected_tools, "retrieved_tools": tool_list}

