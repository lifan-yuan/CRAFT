from utils import *
import datasets
import random
import numpy as np
import openai
import os
import openai
from metric_utils import *
from utils import process_code
def generate_code(example):
    code = forward('codex', prompt=example["question"], input_type="image")
    code = process_code(code)
    return {
        "code": code
    }

from retrieve_tools import retrieve_tool
PROMPT = open(config.codex.prompt).read()

inserted_tools_prompt = """**Note: If necessary, you may also leverage the following tools to directly perform complex operations. 
However, please carefully review the implementation code of the tool functions to determine whether to utilize any of them.
Additionally, consider the appropriate method of passing parameters based on your comprehension of the internal implementation of the tool functions, rather than solely relying on the docstring.**\n"""


def wrap_into_function(func_head, docstring):
    name = func_head.split("(")[0].strip()
    args = ", ".join([arg.split(":")[0].strip() for arg in func_head.split("(")[1].split(")")[0].split(",")])
    return f"def {func_head}:" + "\n" + f"\t'''{docstring}\n\t'''" + "\n" +  f"\treturn {name}({args})\n"

def wrap_into_incontext_sample(query, call):
        code = f"Query: {query}" + "\n" + "def execute_command(image):" + "\n" + "\timage_patch = ImagePatch(image)" + "\n" + f"\treturn {call}\n"
        return code

def count_args_from_call(call):
    record = [] # record all (), [], {}

    tuples = re.findall(r"\((.*?)\)", call)
    if len(tuples) > 1: # first one is the total args
        for i in range(1, len(tuples)):
            record.append(tuples[i])
    
    lists = re.findall(r"\[(.*?)\]", call)
    for i in range(0, len(lists)):
        record.append(lists[i])

    dicts = re.findall(r"\{(.*?)\}", call)
    for i in range(0, len(dicts)):
        record.append(dicts[i])

    # now replace all comma in record with ";" for protection
    for i, sub_string in enumerate(record):
        call = call.replace(sub_string, sub_string.replace(",", ";"))

    # now count the number of args by splitting with ","
    try:
        args = re.findall(r"\((.*?)\)", call)[0].split(", ")
    except:
        print(call, re.findall(r"\((.*?)\)", call))
        exit()
    return len(args)
    
def remove_extra_functions(code, all_tools, retrieved_tools):
    # extract function name and args from retrieved tools
    try:
        function_names = [extract_function_name(item) for item in [all_tools[i] for i in retrieved_tools]]
    except IndexError:
        print(len(all_tools), retrieved_tools)
        exit()
    num_args = [count_args(item) for item in [all_tools[i] for i in retrieved_tools]]
    # extract function name and args from code
    tool_call = set() # may use multiple tools or each using multiple times
    for line in code.split("\n"):
        for func_name in function_names:
            if func_name in line:
                tool_call.add((func_name, line.strip()))

    tool_call = list(tool_call)
    num_args_in_code = []
    for func_name, call in tool_call:
        arg_list = []     
        if "(" in call and ")" in call: # make sure there are args
            num_args_in_code.append((func_name, count_args_from_call(call)))
    filtered_tools = []
    for i, (func_name, num_arg) in enumerate(zip(function_names, num_args)):
        if (func_name, num_arg) in num_args_in_code:
            filtered_tools.append(retrieved_tools[i]) # list[int]
    return filtered_tools


def generate_code_with_retrieval(example, vector_library, model, tokenizer):

    print()
    print(example["question"])
    
    retrieval_results = retrieve_tool(example, vector_library, model, tokenizer, 3)
    retrieved_tools = retrieval_results["retrieved_tools"]
    
    top_k = 3
    while True:
        try:
            tools = retrieved_tools[:top_k]
            if len(tools) > 0:
                inserted_tools = inserted_tools_prompt + "\n" + "\n\n".join([toolbase["tool"][tool] for tool in tools])
                base_prompt = PROMPT.replace("INSERT_TOOL_HERE", inserted_tools) 
            else:
                base_prompt = PROMPT.replace("INSERT_TOOL_HERE", "") 
                inserted_tools = ""

            code = forward('codex', prompt=example["question"], input_type="image", base_prompt=base_prompt)
            code = process_code(code)
        except openai.error.InvalidRequestError as e: # exceed max token length
            print(e)
            top_k -= 1
            continue
        else:
            print()
            print("\n\n".join([toolbase["tool"][tool] for tool in tools]))
            break

    # # write base_prompt to temp.txt
    # with open("temp.txt", "w") as f:
    #     f.write(base_prompt)

    print()
    print(example["question"])
    print(code)
    print("\n"*3)

    return {
        "code": code,
        "inserted_tool_prompts": inserted_tools,
        "retrieved_tools": tools
    }


def validate_results(dataset):

    code_results_path = f"./results/eval/{args.eval_dataset}/{args.model}.json" if not args.retrieval else \
                f"./results/eval/{args.eval_dataset}/{args.model}_retrieval.json"

    if os.path.exists(code_results_path):
        import json
        with open(code_results_path, "r") as f:
            results = json.load(f)
        
        keys = list(range(len(results["prediction"].keys())))
        predictions = [results["prediction"][str(i)] for i in keys]
        groundtruths = [results["groundtruth"][str(i)] for i in keys]
    else:
        global flag
        if not flag:
            init_vision_models()
            flag = True

        predictions = []
        groundtruths = []
        for data in tqdm(dataset):
            image = load_image(data["image_path"])

            # wrap tools into code
            if args.retrieval:
                # should deduplicate the retrieved tools
                retrieved_tools = remove_extra_functions(code, toolbase["tool"], data["retrieved_tools"])
                retrieved_tools = [toolbase["tool"][i] for i in retrieved_tools]
                explanations = [extract_function_docstring(item)[0] for item in retrieved_tools]
                retrieved_tools = [tool.replace(explanation, "") for tool, explanation in zip(retrieved_tools, explanations)]
                # print(retrieved_tools)
                code = "\n\n".join([
                    *retrieved_tools,
                    code
                ])
            
            # execute code
            if code is None:
                prediction = ""
            elif ("pixel" in code) or ("input(" in code) or ("return" not in code): # infinite loop or no return
                print("Error in turbo-generated code.")
                prediction = ""
            else:
                try: # normal cases
                    print(code)
                    prediction = execute_code(code, image)
                    print()
                    print(data["question"])
                    print(prediction, data["answers"])
                except:
                    print("Error in turbo-generated code. ")
                    prediction = ""

            # process bool to yes/no
            if str(prediction) == "True":
                prediction = "yes"
            elif str(prediction) == "False":
                prediction = "no"

            predictions.append(prediction)
            groundtruths.append(data["answers"])

        # save to csv, using pandas
        print("save")
        import pandas as pd
        df = pd.DataFrame({"prediction": predictions, "groundtruth": groundtruths})
        df.to_json(code_results_path, indent=4)
        print("saved")

    return predictions, groundtruths



if __name__ == "__main__":
    # add args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--tool_epoch", type=int, default=-1)
    parser.add_argument("--ablation", type=str, default="none")

    args = parser.parse_args()

    print(os.getenv('CONFIG_NAMES', None))

    dataset = datasets.load_from_disk(f"./datasets/eval/{args.eval_dataset}")#.select(range(1))
    assert len(dataset) == 1000
    flag = False
    ####################################### Code Gen #######################################

    reserved_columns = ["image_id", "image_path", "question", "answers", "code"]
    if args.retrieval:
        reserved_columns  = reserved_columns +  ["retrieved_tools"] 
    all_columns = list(dataset.features.keys())
    
    ##### Tool Scaling #####
    if args.retrieval:
        code_results_path = f"./results/eval/{args.eval_dataset}_{args.model}_retrieval.json"
    else:
        code_results_path = f"./results/eval/{args.eval_dataset}_{args.model}.json"

    ##### Tool-Augmented Code Generation #####
    if args.retrieval:
        ### Prepare model and toolbase for tool retrieval ###
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").cuda()

        ##### initialize toolbase and vector library #####
        toolbase = datasets.Dataset.from_csv('./results/viper/5_deduplicated_tool.csv')
        vector_library = torch.load("./results/viper/vector_library.pt")
        
        print("toolbase length:", len(toolbase))
        
        ### Code Generation with Tool Retrieval ###
        if os.path.exists(code_results_path):
            dataset = datasets.Dataset.from_json(code_results_path)
        else:
            flag = True
            init_vision_models()
            dataset = dataset.map(lambda x: generate_code_with_retrieval(x, vector_library, model, tokenizer), load_from_cache_file=False).remove_columns(set(all_columns)-set(reserved_columns))
            dataset.to_json(code_results_path)

    else: ##### Vinalla Code Generation #####
        if os.path.exists(code_results_path):
            dataset = datasets.Dataset.from_json(code_results_path)
        else:
            flag = True
            init_vision_models()
            dataset = dataset.map(generate_code, load_from_cache_file=False).remove_columns(set(all_columns)-set(reserved_columns))
            dataset.to_json(code_results_path)
    
    
    ####################################### Validate #######################################

    dataset = dataset.map(lambda x: {"image_path": os.path.join(f"./datasets/eval/{args.eval_dataset}/images", x['image_path'])})

    ##### Tool Scaling #####
    if args.retrieval:
        exec_results_path = f"./results/eval/{args.eval_dataset}_{args.model}_retrieval.json"
    else:
        exec_results_path = f"./results/eval/{args.eval_dataset}_{args.model}.json"


    ##### Call Vision Models #####
    predictions, groundtruths = validate_results(dataset)


    ##### Compute Metrics #####
    import pandas as pd
    vqa_acc = 100.00 * compute_vqa_acc(predictions, groundtruths)
    f1 = 100.00 * compute_f1(predictions, groundtruths)
    print(f"Soft accuracy: {vqa_acc}")
    print(f"F1 score: {f1}")

    ##### Write Metrics #####
    os.makedirs(f"./results/metrics/{args.eval_dataset}", exist_ok=True)
    df = pd.DataFrame({"soft_acc": [vqa_acc], "f1": [f1]})
    metric_results_path = f"./results/metrics/{args.eval_dataset}/{args.model}.csv"
    df.to_csv(metric_results_path)
    
    
