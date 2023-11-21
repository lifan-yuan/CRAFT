import openai
import subprocess
import requests
import xmltodict

WolframAlpha_Key = "{Wolfram_Key}"
key_pool = []
f = open("api.key", "r")
lines = f.readlines()
f.close()

for line in lines:
    key_pool.append(line.strip())
key_num = len(key_pool)
openai.api_key = key_pool[0]


# Use ChatGPT API provided by OpenAI
def chat_api(model, instr, current_key, system_msg, temperature=0):
    # print("~~~ In ChatGPT ~~~")
    try_num = 0
    while try_num < 10:
        try:
            # print(instr)
            openai.api_key = key_pool[(try_num+current_key) % key_num]
            response = openai.ChatCompletion.create(
                model=model,
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": instr},
                ],
                temperature=temperature,
                max_tokens=1024,
                top_p=0.7,
            )
            # print(response)
            return response["choices"][0]["message"]["content"].strip()
        except KeyboardInterrupt as e:
            raise e
        except openai.error.InvalidRequestError as e:
            raise e
        except Exception as e:
            # print(e)
            pass
    raise Exception("API Timeout")


# Use Text-Davinci-003 API provided by OpenAI
def davinci_api(message, current_key, temperature=0.3):
    # print("~~~ In Davinci ~~~")
    try_num = 0
    while try_num < key_num:
        try_num += 1
        try:
            openai.api_key = key_pool[(try_num+current_key) % key_num]
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=message,
                temperature=temperature,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response["choices"][0]["text"].strip()
        except KeyboardInterrupt as e:
            raise e
        except:
            print(e)
            pass
    raise Exception("API key exhausted")


# Execute the code in PoT method / during Execution Stage
def process_code(code, is_api=False):
    try:
        all_code_pieces = re.findall(r"```python\n(.*?)```", code, re.DOTALL)
    except:
        all_code_pieces = []
        while "```python" in code:
            st = code.index("```python") + len("```python")
            ed = code.index("```", st) if "```" in code[st:] else len(code)
            all_code_pieces.append(code[st:ed])
            code = code[:st] + code[ed+3:] if "```" in code[st:] else code[:st]

    code_pieces = []
    for i, code_piece in enumerate(all_code_pieces):

        if not is_api:

            if not (code_piece.startswith("import") or \
                code_piece.startswith("from") or \
                code_piece.startswith("def")):
                continue

            code_piece = code_piece.split("\n")

            filtered_code_piece = []
            for line in code_piece: # remove unexpected tool call
                if len(line) == 0: # remain empty line
                    filtered_code_piece.append(line)
                    continue
                if len(line.lstrip())==len(line):
                    if line.startswith("import") or line.startswith("from") or line.startswith("def"):
                        block=True
                        filtered_code_piece.append(line)
                    else:
                        block = False
                if block and len(line.lstrip())!=len(line):
                    filtered_code_piece.append(line)
            code_pieces.append("\n".join(filtered_code_piece).strip())
        else:
            code_piece = code_piece.strip().split("\n")
            if "print" not in code_piece[-1]:
                code_piece[-1] = "print(" + code_piece[-1] + ")"
            code_pieces.append("\n".join(code_piece).strip())
            
    if len(code_pieces) == 0:
        
        if is_api:
            code = code.strip().split("\n")
            if "print" not in code[-1]:
                code[-1] = "print(" + code[-1] + ")"
            code = "\n".join(code).strip()
        else:
            # check what it is, code block or text
            if_succ, _ = execute_code(code)
            if not if_succ:
                code = code.strip().split("\n")
                code = ["# " + line for line in code] # comment out the code to avoid bugs caused by texts
                code = "\n".join(code).strip()
        return code
    else:
        code = "\n\n".join(code_pieces)
    return code
    


def execute_code(code, api_call=None, code_file="code_exec/tmp0"):

    f = open(f"{code_file}.py", "w")
    if api_call is not None:
        code = code + "\n\n" + api_call
    code = code.split("\n")

    f.write("\n".join([
        "import math",
        "from math import *",
        "import spicy",
        "from spicy import *",
        "import numpy as np",
        "import sympy",
        "from sympy import *",
        "import pandas as pd",
        "from collections import *",
        "\n"
    ]))

    f.write("\n".join(code))

    f.close()
    i = 0
    while i < 3:
        try:
            result = subprocess.run(
                ['python', f'{code_file}.py'], capture_output=True, check=False, text=True, timeout=2)
        except Exception as e:
            if code_file in str(e):
                i += 1
                continue
            else:
                return False, e
        else:
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif code_file in m:
                        st = m.index('"/') + 1
                        ed = m.index(f'/{code_file}.py') + 1
                        clr = m[st:ed]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()
            else:
                output = result.stdout
                return True, output.strip()
            break
    return False, "Code run time out!"


import numpy as np
import torch
from torch.utils.data import DataLoader
def compute_simcse(model, tokenizer, texts):
    '''
    Given a list of texts, compute the similarity between each pair of texts.
    :param texts: a list of text.
    :return: a list of similarity scores.
    '''
    data_loader = DataLoader(texts, shuffle=False, batch_size=16)
    embeddings = []
    for batch in data_loader:
        # Tokenize input texts
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        # Get the embeddings
        with torch.no_grad():
            embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.detach().cpu().numpy()
            embeddings.extend(embedding)
    
    # compute similarity
    embeddings = torch.from_numpy(np.array(embeddings))
    del tokenizer, model, data_loader
    return embeddings

def sort_by_similarity(model, tokenizer, current_solutions, ori_dataset):
    
    current_solution_embeddings = compute_simcse(model, tokenizer, current_solutions["question"])
    instruction_embeddings = compute_simcse(model, tokenizer, ori_dataset["question"])

    similarity_matrix = torch.zeros((len(current_solution_embeddings), len(instruction_embeddings)))
    for i in range(len(instruction_embeddings) // 1000 + 1):
        start, end = i*1000, min((i+1)*1000, len(instruction_embeddings))
        part_of_instruction_embeddings = instruction_embeddings[start:end]
        similarity_matrix[:, start:end] = torch.nn.functional.cosine_similarity(current_solution_embeddings.unsqueeze(1).cuda(), part_of_instruction_embeddings.unsqueeze(0).cuda(), dim=2).cpu()
    print(similarity_matrix)
    # for each sample in instruction dataset, find the highest similarity value in current solutions. The correspoding index does not matter.
    similarity_scores, _ = torch.max(similarity_matrix, dim=0)
    print(similarity_matrix.shape, similarity_scores.shape) # similarity_scores.shape == (len(ori_dataset),)
    # sort similarity_scores in ascending order
    sorted_similarity_scores, sorted_similarity_indices = torch.sort(similarity_scores, dim=0)
    print(sorted_similarity_scores[:5], sorted_similarity_scores[-5:])
    # sort bootstrap dataset by sorted_similarity_scores
    sorted_instruction_dataset = ori_dataset.select(sorted_similarity_indices)
    return sorted_instruction_dataset, sorted_similarity_scores, sorted_similarity_indices


### deduplication ###
def extract_function_name(function):
    '''
    Given a python function, use rule-based method to extract the function name.
    :param function: a python function described in string.
    :return: the function name.
    '''
    
    function = function.strip()
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    function = function.strip()
    function_name = function.split("(")[0].strip()
    return function_name

def extract_function_head(function):
    '''
    Given a python function, use rule-based method to extract the function name.
    :param function: a python function described in string.
    :return: the function name.
    '''
    function = function.strip().split("\n")
    function = [func for func in function if "def" in func][0]
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    function_head = function.strip()
    return function_head

def count_args(function_head):
    '''
    Given a python function head, count the number of arguments.
    :param function_head: a python function head.
    :return: the number of arguments.
    '''
    function_head = function_head.strip()
    if function_head.endswith(")"):
        function_head = function_head[:-1]
    if "(" in function_head:
        args = function_head.split("(")[1].strip()
        if args == "":
            return 0
        else:
            return len(args.split(","))
    else:
        return 0

def extract_function_docstring(function):
    '''
    Given a python function, use rule-based method to extract the function docstring.
    :param function: a python function described in string.
    :return:
    '''
    function = function.strip()
    if function.startswith("def"):
        function = function[3:]
    if function.endswith(":"):
        function = function[:-1]
    # return function
    if '"""' in function:
        items = function.split('"""')
    else:
        assert "'''" in function, print(function)
        items = function.split("'''")

    docstring = items[1].strip()
    explanation = docstring.split("\n")[0].strip()
    return (explanation, docstring)

deduplication_template = """Here are several tools with similar functionalities. Your task is to select the most generic one, which can be widely applied and frequently reused across various scenarios. Your decision should be based on your understanding of typical use cases of VQA tasks and the capabilities of the tools, rather than relying on superficial patterns like the frequency of tool names.

Tips: 
1. Consider the level of specificity of the strings in the code to assess its generalizability.
2. Evaluate the tool's functionalities and options to determine its versatility.

### Format ###
Tools are listed below:

No. 0:
Tool 1

No. 1:
Tool 2

...

No. N:
Tool N

Please respond with the numeric number preceding the most general tool using just one token, e.g.: N


### Input ###
The tools are:

{}

Please provide your answer by entering the numeric number preceding the most general tool using only one token: """



import re
import random
def deduplicate_by_chatgpt(tool_list):

    # random.shuffle(tool_list)

    if len(tool_list) > 5: # devide; otherwise, contexts will be too long
        tool_sublists = [tool_list[i:i+5] for i in range(0, len(tool_list), 5)]
        tool_list = []
        for tool_sublist in tool_sublists:
            tool_sublist = ["No. {}:\n{}\n\n".format(i, tool) for i, tool in enumerate(tool_sublist)]
            prompt = deduplication_template.format("\n\n".join(tool_sublist))

            reply = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0613",
                        messages = [
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                        max_tokens=1,
                    )["choices"][0]["message"]["content"].strip()
            
            print("reply: {} / {}".format(reply, set([i for i in range(len(tool_sublist))])))
            reply = re.findall(r"\d+", reply) 
            assert len(reply) <= 1
            reply = int(reply[0]) if len(reply) ==1 else 0 # if all tools are not good, simply choose the first one

            tool_list.append(tool_sublist[reply])

    tool_list = ["No. {}:\n{}\n".format(i, tool) for i, tool in enumerate(tool_list)]
    prompt = deduplication_template.format("\n\n".join(tool_list))
    reply = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages = [
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=1,
                )["choices"][0]["message"]["content"].strip()

    print("reply: {} / {}".format(reply, set([i for i in range(len(tool_list))])))
    reply = re.findall(r"\d+", reply)
    assert len(reply) <= 1
    reply = int(reply[0]) if len(reply) ==1 else 0
    
    return reply


def deduplicate_by_name(all_tools, function_names, function_heads, num_args):
    
    category_head = {}
    category_node = {}

    # Print the community assignments
    for id, (name, head, num_arg) in enumerate(zip(function_names, function_heads, num_args)):
        ###########
        num_arg = 1
        ###########
        if name not in category_head.keys():
            category_head[name] = {} # num_arg: function_head
            category_node[name] = {} # num_arg: node
        
        if num_arg not in category_head[name].keys():
            # add {num_arg: head} to the dict
            category_head[name][num_arg] = []
            category_node[name][num_arg] = []
        # do not add duplicate function
        if head not in category_head[name][num_arg]:
            category_head[name][num_arg].append(head)
            category_node[name][num_arg].append(id)

    # sort category by key (which is a number)
    category_head = {k: v for k, v in sorted(category_head.items(), key=lambda item: item[0])}

    # flatten category_node to list
    for name in category_node.keys():
        for num_arg in category_node[name].keys():
            if len(category_node[name][num_arg]) == 1:
                category_node[name][num_arg] = category_node[name][num_arg][0] # flatten
                continue
            most_general = deduplicate_by_chatgpt([all_tools[i] for i in category_node[name][num_arg]])
            category_node[name][num_arg] = category_node[name][num_arg][most_general]
            category_head[name][num_arg] = [category_head[name][num_arg][most_general]]

    # flatten the category to category[community_id] = [function_head1, function_head2, ...]
    category_head = {k: [sublist[0] for sublist in v.values()] for k, v in category_head.items()} 
    
    # flatten category_node to list
    category_node = [id for ids_with_same_name in category_node.values() for id in ids_with_same_name.values()]
    
    # verification
    heads = [function_heads[i] for i in category_node]
    flatted_category_head = [head for sublist in category_head.values() for head in sublist]
    assert set(heads) == set(flatted_category_head), print(len(heads), len(flatted_category_head), set(heads)-set(flatted_category_head))
    
    return category_head, category_node
