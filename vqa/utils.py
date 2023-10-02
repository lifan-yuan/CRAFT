import requests
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import transformers
from typing import *

# viper gpt
from rich.console import Console
from rich.syntax import Syntax
from rich.live import Live
from rich.padding import Padding
import ast
import astunparse


# viper gpt
from image_patch import *
# from video_segment import *
from vision_models import *
from vision_processes import *

from prompts.utils_prompt import *
# must import time here, since other modules have already imported time from time 
import time

console = Console(highlight=False, force_terminal=False)

gpt3_model = GPT3Model()



def compute_simcse(model, tokenizer, texts):
    '''
    Given a list of texts, compute the similarity between each pair of texts.
    :param texts: a list of text.
    :return: a list of similarity scores.
    '''
    data_loader = DataLoader(texts, shuffle=False, batch_size=32)
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

def sort_by_similarity(included_embeddings, remaining_embeddings):

    similarity_matrix = torch.zeros((len(included_embeddings), len(remaining_embeddings)))

    for i in range(len(remaining_embeddings) // 1000 + 1): # to avoid oom error
        start, end = i*1000, min((i+1)*1000, len(remaining_embeddings))
        part_of_instruction_embeddings = remaining_embeddings[start:end]
        # compute similarity using matrix multiplication
        part_of_similarity = torch.nn.functional.cosine_similarity(included_embeddings.unsqueeze(1).cuda(), part_of_instruction_embeddings.unsqueeze(0).cuda(), dim=2).cpu()
        similarity_matrix[:, start:end] = part_of_similarity
        assert similarity_matrix[:, start:end].shape == part_of_similarity.shape

    # for each sample in the remaining dataset, find the highest similarity value with any included examples.
    similarity_scores, _ = torch.max(similarity_matrix, dim=0) # index does not matter
    print(similarity_matrix.shape, similarity_scores.shape) # similarity_scores.shape == (len(instruction_dataset),)
    
    # sort similarity_scores in ascending order
    sorted_similarity_scores, sorted_similarity_indices = torch.sort(similarity_scores, dim=0)
    print(sorted_similarity_scores[:5], sorted_similarity_scores[-5:])

    return sorted_similarity_indices, sorted_similarity_scores

### execution utils ###
def load_image(path):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        image = transforms.ToTensor()(image)
    else:
        image = Image.open(path).convert('RGB')
        image = transforms.ToTensor()(image)
    return image

def extract_code(code): # TODO: may have bug
    code_blocks = code.split("\n\n")
    for code_block in code_blocks:
        if "def execute_command" in code_block: # return first one
            code_block = code_block.split("\ndef ") # several functions
            if len(code_block) == 1:
                return code_block[0]
            else:
                return "\n".join([code_block[0], "def "+code_block[1]]) # query + `execute_command` function
    return "\n\n".join(code_blocks)

def split_codeline_and_indent_level(codeline):
    origlen = len(codeline)
    codeline = codeline.lstrip(" ")
    indent = origlen - len(codeline)
    indent = "\t" * int(indent / 4 + 0.5) # '\t' is 4 spaces; rounding off
    return codeline, indent

def process_code(code):
    code = extract_code(code).split("\n")
    newcode = []
    for codeline in code:
        if codeline.startswith(" "):
            codeline, indent = split_codeline_and_indent_level(codeline)
            newcode.append(f"{indent}{codeline}")
        else:
            newcode.append(codeline)
    return "\n".join(newcode)

def execute_code(code, image, question=None):
    code = astunparse.unparse(ast.parse(code))
    console.print(code)
    exec(compile(code, filename='Codex', mode='exec'), globals())
    if question is None:
        result = execute_command(image)  # The code is created in the exec()
    else:
        result = execute_command(image, question)
    # console.rule(f"[bold]Final Result[/bold]", style="chartreuse2")
    
    return result


### evaluation utils ###
def eval_generated_code(code, data):
    
    image_path = os.path.join("./datasets/coco_images/train2017", data["image_path"])
    image = load_image(image_path)
    
    print("question:", data["question"])

    try:
        result = execute_code(code, image)
    except: # there are bugs in the generated code
        print("bugs in the generated code")
        is_correct = -1
        return is_correct, None
    else:
        data = {
            "question": data["question"],
            "prediction": result,
            "groundtruth": data["answer"],
        }
        is_correct = check_consistency(data)

        return is_correct, result

### remove safety texts ###
from nltk import sent_tokenize
chatgpt_filters = ["sorry", "please", "an ai model", "a language model", "an ai language model", "do not have access to", "it is impossible to"]
def process_llm_outputs(text):
    text = [sent for sent in sent_tokenize(text) if not any(phrase in sent.lower() for phrase in chatgpt_filters)]
    if len(text) == 0:
        return ""
    return " ".join(text)


def check_consistency(data):

    question, prediction, groundtruth = str(data["question"]), str(data["prediction"]), str(data["groundtruth"])

    if prediction  == "" or prediction == "None":
        reward = 0
        return reward

    prediction = process_llm_outputs(prediction)
    groundtruth = process_llm_outputs(groundtruth)
    if "ImagePatch" in prediction or prediction.strip() == "": # wrong answer format
        reward = 0
        return reward
    prompt = CONSISTENCY_PROMPT.format(question, prediction, groundtruth)
    print(prompt)

    from image_patch import llm_query
    while True:
        try:
            response = llm_query(prompt)
        except Exception as e:
            print(e)
        else:
            break
    
    # remove all punctuatoin
    if isinstance(response, str):
        response = re.sub(r"[^a-zA-Z0-9]+", "", response.lower())
    print(response)
    reward = 1 if response == "yes" else 0

    return reward


### abstraction ###
def abstraction(query, solution):

    messages=[
        {"role": "user", "content": abstraction_template.format(
                                        query=query, 
                                        solution=solution)},
        ]

    response = gpt3_model.query_with_message(messages, model="gpt-4-0613", max_tokens=1024, temperature=0.0)
    # extract tool and api_call from the response
    tool, api_call = response.split("The example to call the tool is: ")
    tool = tool.split("The final generic tool with docstring is:")[1]
    if "```python" and "```" in tool:
        # extract the code between ```python and ``` using regex
        tool = re.findall(r"```python(.*?)```", tool, re.DOTALL)[0]
    tool = tool.strip()
    api_call = api_call.strip("`").strip()
    return tool, api_call



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
    function = function.strip().split("\n")[0]
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
            messages = [
                {"role": "user", "content": prompt}
            ]
            reply = gpt3_model.query_with_message(messages, max_tokens=5)
            
            # print("message:", messages)
            print("reply: {} / {}".format(reply, set([i for i in range(len(tool_sublist))])))
            reply = re.findall(r"\d+", reply) 
            assert len(reply) <= 1, print(messages, "\n", reply)
            reply = int(reply[0]) if len(reply) ==1 else 0 # if all tools are not good, simply choose the first one

            tool_list.append(tool_sublist[reply])

    tool_list = ["No. {}:\n{}\n".format(i, tool) for i, tool in enumerate(tool_list)]
    prompt = deduplication_template.format("\n\n".join(tool_list))
    messages = [
        {"role": "user", "content": prompt},
    ]
    reply = gpt3_model.query_with_message(messages, model="gpt-3.5-turbo-0613", max_tokens=5, temperature=0.0)
    # print("message:", messages)
    print("reply: {} / {}".format(reply, set([i for i in range(len(tool_list))])))
    reply = re.findall(r"\d+", reply)
    assert len(reply) <= 1, print(messages, "\n", reply)
    reply = int(reply[0]) if len(reply) ==1 else 0
    
    return reply


def deduplicate_by_name(all_tools, function_names, function_heads, num_args):
    
    category_head = {}
    category_tool = {}

    # Print the community assignments
    for id, (name, head, num_arg) in enumerate(zip(function_names, function_heads, num_args)):

        if name not in category_head.keys():
            category_head[name] = {} # num_arg: function_head
            category_tool[name] = {} # num_arg: tool
        
        if num_arg not in category_head[name].keys():
            # add {num_arg: head} to the dict
            category_head[name][num_arg] = []
            category_tool[name][num_arg] = []
        # do not add duplicate function
        if head not in category_head[name][num_arg]:
            category_head[name][num_arg].append(head)
            category_tool[name][num_arg].append(id)

    # sort category by key (which is a number)
    category_head = {k: v for k, v in sorted(category_head.items(), key=lambda item: item[0])}

    # flatten category_tool to list
    for name in category_tool.keys():
        for num_arg in category_tool[name].keys():
            if len(category_tool[name][num_arg]) == 1:
                category_tool[name][num_arg] = category_tool[name][num_arg][0] # flatten
                continue
            most_general = deduplicate_by_chatgpt([all_tools[i] for i in category_tool[name][num_arg]])
            category_tool[name][num_arg] = category_tool[name][num_arg][most_general]
            category_head[name][num_arg] = [category_head[name][num_arg][most_general]]

    # flatten the category to category[community_id] = [function_head1, function_head2, ...]
    category_head = {k: [sublist[0] for sublist in v.values()] for k, v in category_head.items()} 
    
    # flatten category_tool to list
    category_tool = [id for ids_with_same_name in category_tool.values() for id in ids_with_same_name.values()]
    
    # verification
    heads = [function_heads[i] for i in category_tool]
    flatted_category_head = [head for sublist in category_head.values() for head in sublist]
    assert set(heads) == set(flatted_category_head), print(len(heads), len(flatted_category_head), set(heads)-set(flatted_category_head))
    
    return category_head, category_tool