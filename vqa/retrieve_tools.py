import pandas as pd
import numpy as np
import datasets
import openai
from utils import *


def planning(query):    

    messages = [
        {"role": "user", "content": retrieval_template.format(query=query)},
    ]
    response = gpt3_model.query_with_message(messages, max_tokens=200)
    plans = [query, response.split("The final answer is: ")[1].strip()]

    try:
        expected_tools = eval(response.split("\n\n")[-2].split("The useful functions are: ")[1].strip("."))
    except:
        expected_tools = eval(response.split("\n\n")[-2].split("The useful function is: ")[1].strip("."))

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
    
    ##### Retrieval Stage of CRAFT #####

    # decompose the query into sub-tasks
    plans, expected_tools = planning(example['query'])
    plan_embeddings = compute_simcse(model, tokenizer, plans)
    expected_tool_embeddings = compute_simcse(model, tokenizer, expected_tools)

    # match plan with tools from different perspectives
    tool_by_explanation = match_plan_from_single_perspective(plan_embeddings[1:], vector_library["explanation_embedding"], k=10)
    tool_by_name = match_plan_from_single_perspective(expected_tool_embeddings, vector_library["name_embedding"], k=5)
    tool_by_query = match_plan_from_single_perspective(plan_embeddings[0].unsqueeze(0), vector_library["query_embedding"], k=10)

    counter = Counter([ 
                        *[item for sublist in tool_by_explanation for item in sublist], # k_1*len(plans)
                        *[item for sublist in tool_by_name for item in sublist], # k_1*len(plans)
                        *[item for sublist in tool_by_query for item in sublist], # k_1*1
                    ])

    top_k = counter.most_common(k) 
    tool_list.extend([tool for (tool, count) in top_k if count >= 2]) # must at least have 2 votes

    tool_list = list(set(tool_list))
    return {"expected_tools": expected_tools, "retrieved_tools": tool_list}

