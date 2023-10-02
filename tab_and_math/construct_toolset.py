import os
import json
import re
from utils import *
from tqdm import tqdm
import datasets
from datasets import Value
import pandas as pd
from grader import grade_answer
from transformers import AutoTokenizer, AutoModel
# ================================ #


def validate_code_solution(example):

    # validate solution
    if "tool" in example.keys():
        code, api_call = example["tool"], example["tool_call"]
    else:
        code, api_call = example["code"], example["api_call"]
    code = process_code(code)
    api_call = process_code(api_call, is_api=True)

    if_succ, info = execute_code(code, api_call, code_file)
    if if_succ: 
        print("~~~ Runing successfully ~~~")

        print("info:", info, "answer:", example["answer"])
        is_correct = grade_answer(str(info), str(example["answer"]))
        if not is_correct:
            print("!!! Wrong answer !!!")
            print("Let's check original code and api_call!")
            print(code)
            print(api_call)
            _, info = execute_code(code, api_call, code_file)
            print("info:", info, "original answer:", example["answer"])
            print("\n"*3)
        return is_correct
    else:
        print("info:", info)

    return False




def specific_solution_generation(example):

    f = open(prompt_creation, "r")
    prompt1 = f.read().strip()
    f.close
    
    f = open(prompt_decision, "r")
    prompt2 = f.read().strip()
    f.close

    # generate solution
    env1 = prompt1.replace("===qst===", example["question"])
    if task == "TabMWP":
        env1 = env1.replace("===table===", example["table"])
    response1 = gen_func("gpt-4-0613", env1, start_key, sys_msg, temperature=temperature)

    # generate api call
    env2 = prompt2.replace("===qst===", example["question"]).replace("===tool===", response1)
    if task == "TabMWP":
        env2 = env2.replace("===table===", example["table"])

    response2 = gen_func("gpt-4-0613", env2, start_key, sys_msg, temperature=temperature)
    
    # To further enhance the validity of the tool format
    if "```python\n" not in response1:
        response1 = "```python\n" + response1 + "\n```"

    if "```python\n" not in response2:
        response2 = "```python\n" + response2 + "\n```"

    response1 = process_code(response1)
    response2 = process_code(response2, is_api=True)

    return {"code": response1,
            "api_call": response2}




def abstract_tools(example):

    f = open(prompt_abstraction, "r")
    prompt = f.read().strip()
    f.close

    # generate solution

    env = prompt.replace("===qst===", example["question"])
    if task == "TabMWP":
        env = env.replace("===table===", example["table"])

    env = env.replace("===specific solution===", "\n\n".join([example["code"], example["api_call"]])).strip()#.replace(function_name, "execute_command").strip())
    response = gen_func("gpt-4-0613", env, start_key, sys_msg, temperature=temperature)

    # To further enhance the validity of the tool format
    if "```python\n" in response:
        response  = re.findall(r"```python(.*?)```", response, re.DOTALL)[0]

    # extract tool and tool_call from the response
    tool, tool_call = response.split("# The example to call the tool is:")
    if "# The final generic tool with docstring is:" in tool:
        tool = tool.split("# The final generic tool with docstring is:")[1]
    if "```python" and "```" in tool:
        # extract the code between ```python and ``` using regex
        tool = re.findall(r"```python(.*?)```", tool, re.DOTALL)[0]
    tool = process_code(tool.strip())
    tool_call = process_code(tool_call.strip("`").strip(), is_api=True)

    return {"tool": tool, "tool_call": tool_call}



def deduplicate_tools(all_tools, save_dir_path):

    if os.path.exists(f"{save_dir_path}/5_deduplicated_tools.csv"):
        deduplicated_tools = datasets.Dataset.from_csv(f"{save_dir_path}/5_deduplicated_tools.csv")
    else:
        tool_list = all_tools["tool"]

        selected_tools = []

        function_names = [" ".join(extract_function_name(item).split()) for item in tool_list]
        function_heads = [extract_function_head(item) for item in tool_list]
        num_args = [count_args(item) for item in function_heads]

        category_head, category_node = deduplicate_by_name(tool_list, function_names, function_heads, num_args)     

        # write to json
        with open(f"{save_dir_path}/category_head.json", "w") as f:
            json.dump(category_head, f, indent=4)
        
        deduplicated_tools = all_tools.select(category_node)
        deduplicated_tools.to_csv(f"{save_dir_path}/5_deduplicated_tools.csv")
    return deduplicated_tools



# vector_library = torch.load("./datasets/vector_library.pt")
def construct_vector_library(toolbase, model, tokenizer, save_dir_path):
    all_tools = toolbase["tool"]
    function_names = [" ".join(extract_function_name(item).split("_")) for item in all_tools]
    function_explanations = [extract_function_docstring(item)[0] for item in all_tools]
    function_docstrings = [extract_function_docstring(item)[1] for item in all_tools]
    function_queries = toolbase["question"]


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
                datasets.Dataset.from_json(f"./{save_dir_path}/epoch_{i}/1_raw_specific_solutions.json")
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




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="GSM8K", help="Choose from 'MATH', 'TabMWP', 'GSM8K'")
    args = parser.parse_args()

    # Choose from "MATH", "TabMWP", "Creation"
    task = args.task
    mode = "normal"
    if task == "MATH":
        fields = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        sys_msg = "You are a helpful assistant in answering math competition problems."
    elif task == "TabMWP":
        fields = ["tabmwp"]
        sys_msg = "You are a helpful assistant in answering questions with tabular contents."
    else:
        raise Exception("Wrong task name!")
    # ================================ #     
    start_key = 0
    temperature = 0.3
    prompt_creation = f"{task}/prompt_lib/prompt_CREATOR_creation.md"
    prompt_decision = f"{task}/prompt_lib/prompt_CREATOR_decision.md"
    prompt_rectification = f"{task}/prompt_lib/prompt_rectification.md"
    prompt_abstraction = f"{task}/prompt_lib/prompt_abstraction.md"
    code_file = "code_exec/tmp0"
    gen_func = chat_api
 

    for field in fields[:1]:
        save_file = f"{task}/results/results_{field}_CREATOR.md"
        f = open(save_file, "w")
        f.close()
        
        ori_dataset = datasets.Dataset.from_json(f"{task}/dataset/train/{field}.jsonl")#.select(range(20))
        ori_dataset = ori_dataset.add_column("tool_id", [i for i in range(len(ori_dataset))])
        print(len(ori_dataset))
        save_path_dir = f"{task}/craft/{field}"
        os.makedirs(save_path_dir, exist_ok=True)


        sampled_ids = select_samples(ori_dataset, total_extended_size=500, initial_extended_size=200, extended_size_per_epoch=100, save_dir_path=save_path_dir)

        print({k:len(v) for k,v in sampled_ids.items()})
        for epoch in sampled_ids.keys():
            print("Epoch: ", epoch)

            dataset = ori_dataset.select(sampled_ids[epoch])
            save_path_dir = f"{task}/craft/{field}/epoch_{epoch}"
            os.makedirs(save_path_dir, exist_ok=True)
            # Step 1
            if not os.path.exists(f"{save_path_dir}/1_raw_specific_solutions.json"):
                dataset = dataset.map(specific_solution_generation, desc="Specific Solution Generation")
                dataset.to_json(f"{save_path_dir}/1_raw_specific_solutions.json")
            else:
                dataset = datasets.Dataset.from_json(f"{save_path_dir}/1_raw_specific_solutions.json")
            # Step 2
            if not os.path.exists(f"{save_path_dir}/2_valid_specific_solutions.csv"):
                dataset = dataset.filter(validate_code_solution, desc="Validate Code Solution", load_from_cache_file=False)
                dataset.to_csv(f"{save_path_dir}/2_valid_specific_solutions.csv")
            else:
                dataset = datasets.Dataset.from_csv(f"{save_path_dir}/2_valid_specific_solutions.csv")
            # Step 3
            if not os.path.exists(f"{save_path_dir}/3_all_general_tools.json"):
                dataset = dataset.map(abstract_tools, desc="Tool Abstraction", load_from_cache_file=False)
                dataset.to_json(f"{save_path_dir}/3_all_general_tools.json")
            else:
                dataset = datasets.Dataset.from_json(f"{save_path_dir}/3_all_general_tools.json")
            # Step 4
            if not os.path.exists(f"{save_path_dir}/4_valid_tools.csv"):
                dataset = dataset.filter(validate_code_solution, desc="Validate Abstract Tool", load_from_cache_file=False)
                dataset.to_csv(f"{save_path_dir}/4_valid_tools.csv")
            else:
                dataset = datasets.Dataset.from_csv(f"{save_path_dir}/4_valid_tools.csv")
            # Step 5
            dataset = deduplicate_tools(dataset, save_path_dir)
            
        print("end iteration")
        
        save_path_dir = f"{task}/craft/{field}"
        dataset = datasets.Dataset.from_csv(f"{save_path_dir}/epoch_0/5_deduplicated_tools.csv")
        new_features = dataset.features.copy()
        new_features["answer"] = Value("string")
        dataset = dataset.cast(new_features)
        for epoch in range(1, 4):
            epoch_dataset = datasets.Dataset.from_csv(f"{save_path_dir}/epoch_{epoch}/5_deduplicated_tools.csv")
            new_features = epoch_dataset.features.copy()
            new_features["answer"] = Value("string")
            epoch_dataset = epoch_dataset.cast(new_features)
            dataset = datasets.concatenate_datasets([dataset, epoch_dataset])
        
        dataset = deduplicate_tools(dataset, save_path_dir)
        print(len(dataset))
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").cuda()
        construct_vector_library(dataset, model, tokenizer, save_path_dir)
