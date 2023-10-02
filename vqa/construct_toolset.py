from utils import *
import datasets
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import json

import os
import datasets
import torch
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Step 1. Infer on instruction dataset to obtain specific solutions
def generate_code(example, task):
    code = forward('codex', prompt=example["question"], input_type="image")
    code = process_code(code)

    return {
        "question": example["question"],
        "code": code,
        "task": task
    }


# Step 2. Validate the code
def validate_turbo_code(generated_code, save_path):
    validate_tool = True if "4_valid_tools.csv" in save_path else False
    generated_code.remove_columns(["question"])
    if os.path.exists(save_path):
        generated_code = datasets.Dataset.from_csv(save_path)
    else:
        filtered = []
        for i, data in tqdm(enumerate(generated_code)):
            pred = data["tool_call"] if validate_tool else data["code"]
            assert isinstance(pred, str)
            if "pixel" in pred:
                continue
            reward, prediction = eval_generated_code(pred, data)
            if reward == 1:
                filtered.append(i)
                print("{}/{}".format(len(filtered), len(generated_code)))
        print("filtered: {}/{}".format(len(filtered), len(generated_code)))
        generated_code = generated_code.select(filtered)
        generated_code.to_csv(save_path)
    return generated_code


# Step 3. Abstract the specific solutions to general tools
def abstract_tools(specific_solutions, save_dir_path="./results/viper"):

    if os.path.exists(f"{save_dir_path}/3_all_general_tools.csv"):
        general_tools_with_calls = datasets.Dataset.from_csv(f"{save_dir_path}/3_all_general_tools.csv")
    else:
        reserved_features = ["image_path", "question", "query", "answer", "code", "tool_id"]
        all_features = specific_solutions.features.keys()
        specific_solutions = specific_solutions.remove_columns(set(all_features) - set(reserved_features))
        general_tools = specific_solutions.map(lambda x: {"tool_and_call": abstraction(x["query"], x["code"])})
        general_tools_with_calls = general_tools.map(lambda x: {"tool": x["tool_and_call"][0], "call": x["tool_and_call"][1]}).remove_columns(["tool_and_call"])
        general_tools_with_calls.to_csv(f"{save_dir_path}/3_all_general_tools.csv")
    return general_tools_with_calls


# Step 4. Validate the general tools
def validate_tools(all_tools, save_dir_path="./results/viper"):

    def process_to_function(tool, call):
        code = "\n".join([
            "def execute_command(image):\n",
            "\n".join(["\t"+line for line in tool.split("\n")]),
            "\t" + "image_patch = ImagePatch(image)",
            "\t" + f"return {call}"
        ])
        return code

    if os.path.exists(f"{save_dir_path}/4_valid_tools.csv"):
        valid_tools = datasets.Dataset.from_csv(f"{save_dir_path}/4_valid_tools.csv")
    else:
        all_tools = all_tools.map(lambda x: {"tool_call": "\n".join(["from PIL import Image", "from typing import *", "from image_patch import *", process_to_function(x["tool"], x["call"].strip())])})
        valid_tools = validate_turbo_code(all_tools, f"{save_dir_path}/4_valid_tools.csv")

    return valid_tools


# Step 5. Deduplicate the tools
def deduplicate_tools(all_tools, save_dir_path="./results/viper"):

    if os.path.exists(f"{save_dir_path}/5_deduplicated_tools.csv"):
        deduplicated_tools = datasets.Dataset.from_csv(f"{save_dir_path}/5_deduplicated_tools.csv")
    else:
        tool_list = all_tools["tool"]

        function_names = [" ".join(extract_function_name(item).split()) for item in tool_list]
        function_heads = [extract_function_head(item) for item in tool_list]
        num_args = [count_args(item) for item in function_heads]

        category_head, category_node = deduplicate_by_name(tool_list, function_names, function_heads, num_args)     

        # write to json
        with open(f"{save_dir_path}/category_head.json", "w") as f:
            json.dump(category_head, f, indent=4)
        
        deduplicated_tools = all_tools.select(category_node)
        deduplicated_tools.to_csv(f"{save_dir_path}/5_deduplicated_tool.csv")

    return deduplicated_tools
            

# Check if continue
def check_if_continue(specific_solutions, extended_size_per_epoch, current_save_dir_path):
    if os.path.exists(f"{current_save_dir_path}/4_valid_tools.csv"):
        deduplicated_tools = pd.read_csv(f"{current_save_dir_path}/4_valid_tools.csv")
    else:
        valid_specific_solutions = validate_turbo_code(specific_solutions, save_path=f"{current_save_dir_path}/2_valid_specific_solutions.csv")
        general_tools = abstract_tools(valid_specific_solutions, current_save_dir_path)
        valid_tools = validate_tools(general_tools, current_save_dir_path)
        deduplicated_tools = deduplicate_tools(valid_tools, current_save_dir_path)
    return True


print("set up ssh connection")
def filter_direct_query(query, max_tokens=1):
    print("query:", query)
    query = query.replace('"', "'")
    message = [{"role": "user", "content": filter_template.format(query=query)}]
    response = gpt3_model.query_with_message(message, model="gpt-3.5-turbo-0613", max_tokens=max_tokens, temperature=0.0)
    return 1 if re.sub(r"[^a-zA-Z0-9]+", "", response.lower()) == "no" else 0



def select_samples(ori_dataset, total_extended_size=1000, initial_extended_size=500, extended_size_per_epoch=100, save_dir_path="./results/viper"):
    
    all_solutions = []
    epochs = (total_extended_size - initial_extended_size) // extended_size_per_epoch + 1
    if os.path.exists(f"{save_dir_path}/sampled_ids.pt"):
        sampled_ids = torch.load(f"{save_dir_path}/sampled_ids.pt")
        if len(sampled_ids.keys()) == epochs and len(sampled_ids[0]) == initial_extended_size and all([len(sampled_ids[i]) == extended_size_per_epoch for i in range(1, epochs)]):
            print("We have sampled all the epochs, reuse the ids.")
            return sampled_ids
        elif len(sampled_ids.keys()) < epochs and len(sampled_ids[0]) == initial_extended_size and all([len(sampled_ids[i]) == extended_size_per_epoch for i in range(1, len(sampled_ids.keys()))]):
            print("We have sampled the first epochs, but not all. Continue sampling.")
            sampled_ids = sampled_ids
            all_solutions = datasets.concatenate_datasets([
                datasets.Dataset.from_json(f"{save_dir_path}/epoch_{i}/1_raw_specific_solutions.json")
                for i in range(len(sampled_ids.keys()))
            ])
        else:
            print("expected:", epochs, total_extended_size, initial_extended_size, extended_size_per_epoch)
            print("actual:", {k:len(v) for k,v in sampled_ids.items()})
            # exit()
            print("The sampled_ids file is not correct, delete it and rerun the code.")
            sampled_ids = {}
    else:
        sampled_ids = {}

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").cuda()

    for epoch in range(len(sampled_ids.keys()), epochs):
        ori_dataset = ori_dataset.shuffle()

        current_save_dir_path = f"{save_dir_path}/epoch_{epoch}"
        os.makedirs(current_save_dir_path, exist_ok=True)
        if epoch == 0:
            sampled_dataset = ori_dataset.select(range(initial_extended_size))
        else:
            sorted_instruction_dataset, sorted_similarity_scores, sorted_similarity_indices = \
                                    sort_by_similarity(model, tokenizer, all_solutions, ori_dataset)
            sampled_dataset = sorted_instruction_dataset.select(range(extended_size_per_epoch))
            print(len(sampled_dataset))

        sampled_ids[epoch] = sampled_dataset["tool_id"]
        all_solutions = datasets.concatenate_datasets([all_solutions, sampled_dataset]) if epoch != 0 \
                        else sampled_dataset
        ori_dataset = ori_dataset.filter(lambda x: x["tool_id"] in list(set(ori_dataset["tool_id"]) - set(all_solutions["tool_id"])))

    torch.save(sampled_ids, f"{save_dir_path}/sampled_ids.pt")
    del model, tokenizer
    assert len(sampled_ids[0]) == initial_extended_size and all([len(sampled_ids[i]) == extended_size_per_epoch for i in range(1, epochs)])
    return sampled_ids


def construct_toolbase(instruction_dataset, total_extended_size, initial_extended_size, extended_size_per_epoch, save_dir_path="./results/viper"):
    
    if os.path.exists(f"{save_dir_path}/5_deduplicated_tool.csv"):
        toolbase = datasets.Dataset.from_csv(f"{save_dir_path}/5_deduplicated_tool.csv")
    else:

        sampled_ids = select_samples(instruction_dataset, total_extended_size, initial_extended_size, extended_size_per_epoch, save_dir_path)
        epochs = (total_extended_size - initial_extended_size) // extended_size_per_epoch + 1
        for epoch in range(epochs):
            print("epoch:", epoch)
            current_save_dir_path = f"{save_dir_path}/epoch_{epoch}"
            os.makedirs(current_save_dir_path, exist_ok=True)
            if os.path.exists(f"{current_save_dir_path}/1_raw_specific_solutions.json"):
                print("load")
                specific_solutions = datasets.Dataset.from_json(f"{current_save_dir_path}/1_raw_specific_solutions.json")
            else:
                sampled_dataset = instruction_dataset.filter(lambda x: x["tool_id"] in sampled_ids[epoch])
                specific_solutions = sampled_dataset.map(lambda x: generate_code(x, "vqa"))
                specific_solutions.to_json(f"{current_save_dir_path}/1_raw_specific_solutions.json")
            
            global flag
            if not flag:
                init_vision_models()
                flag = True
            if os.path.exists(f"{current_save_dir_path}/4_valid_tools.csv"):
                deduplicated_tools = pd.read_csv(f"{current_save_dir_path}/4_valid_tools.csv")
            else:
                valid_specific_solutions = validate_turbo_code(specific_solutions, save_path=f"{current_save_dir_path}/2_valid_specific_solutions.csv")
                general_tools = abstract_tools(valid_specific_solutions, current_save_dir_path)
                valid_tools = validate_tools(general_tools, current_save_dir_path)
                deduplicated_tools = deduplicate_tools(valid_tools, current_save_dir_path)

        toolbase = datasets.concatenate_datasets([datasets.Dataset.from_csv(f"{save_dir_path}/epoch_{epoch}/4_valid_tools.csv") for epoch in range(epochs)])
        toolbase = deduplicate_tools(toolbase, save_dir_path)
    return toolbase


def construct_vector_library(model, tokenizer, toolbase, save_dir_path):
    
    if os.path.exists(f"{save_dir_path}/vector_library.pt"):
        vector_library = torch.load(f"{save_dir_path}/vector_library.pt")
    else:
        all_tools = toolbase["tool"]
        function_names = [" ".join(extract_function_name(tool).split("_")) for tool in all_tools]
        function_explanations = [extract_function_docstring(tool)[0] for tool in all_tools]
        function_docstrings = [extract_function_docstring(tool)[1] for tool in all_tools]
        function_queries = toolbase["query"]


        name_embedding = compute_simcse(model, tokenizer, function_names)
        explanation_embedding = compute_simcse(model, tokenizer, function_explanations)
        docstring_embedding = compute_simcse(model, tokenizer, function_docstrings)
        query_embedding = compute_simcse(model, tokenizer, function_queries)

        # save as a dict
        vector_library = {
            "name_embedding": name_embedding,
            "explanation_embedding": explanation_embedding,
            "docstring_embedding": docstring_embedding,
            "query_embedding": query_embedding
        }

        torch.save(vector_library, f"{save_dir_path}/vector_library.pt")

    return vector_library

flag = False
if __name__ == "__main__":
    vqa_dataset = datasets.Dataset.from_json('./datasets/vqa_dataset.json')
    llava_dataset = datasets.Dataset.from_json("./datasets/llava.json")

    reserved_columns = ["image_path", "question", "instruction", "answer", "id"]

    if "tool_id" not in vqa_dataset.features.keys():
        vqa_dataset = vqa_dataset.add_column("tool_id", ["vqa_" + str(i) for i in range(len(vqa_dataset))]).remove_columns(set(vqa_dataset.features.keys())-set(reserved_columns))
        llava_dataset = llava_dataset.add_column("tool_id", ["llava_" + str(i) for i in range(len(llava_dataset))]).remove_columns(set(llava_dataset.features.keys())-set(reserved_columns))
        vqa_dataset.to_json("./datasets/vqa_dataset.json")
        llava_dataset.to_json("./datasets/llava.json")

    
    instruction_dataset = datasets.concatenate_datasets([vqa_dataset, llava_dataset])

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").cuda()

    toolbase = construct_toolbase(instruction_dataset, total_extended_size=2000, initial_extended_size=1000, extended_size_per_epoch=100, save_dir_path="./results/viper_ablation")
    vector_library = construct_vector_library(model, tokenizer, toolbase, "results/viper_ablation")
    print(len(toolbase["tool"]), vector_library["name_embedding"].shape)