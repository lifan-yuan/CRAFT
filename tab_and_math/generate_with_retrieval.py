import json
import re
import os
from utils import *
from retrieve_tools import *
from tqdm import tqdm
from grader import grade_answer

import math
from math import *
import numpy as np
import pandas as pd
from collections import *
import threading




def inference(data):
    
    env1 = prompt1.replace("===qst===", data["question"])
    if task == "TabMWP":
        env1 = env1.replace("===table===", data["table"])

    ### retrieval ###
    if retrieval:
        retrieved_tools = retrieve_tool(data, vector_library, model, tokenizer)["retrieved_tools"]
        
        top_k = 3
        while top_k > 0:
            try:
                env1 = prompt1.replace("===qst===", data["question"])
                if task == "TabMWP":
                    env1 = env1.replace("===table===", data["table"])
                    
                tools = retrieved_tools[:top_k]
                if len(tools) > 0:
                    inserted_tools = "\n\n".join([toolbase["tool"][tool] for tool in tools])
                else:
                    inserted_tools = "None"
                
                env1 = env1.replace("===retrieved tools===", inserted_tools)
                response1 = gen_func("gpt-3.5-turbo-0613", env1, start_key, sys_msg, temperature=temperature)
                
            except openai.error.InvalidRequestError as e: # exceed max token length
                # print(e)
                top_k -= 1
                continue
            else:
                print()
                for tool in tools:
                    print(extract_function_head(toolbase["tool"][tool]))
                break
    #################
    else:
        env1 = env1.replace("===retrieved tools===", "")
        response1 = gen_func("gpt-3.5-turbo-0613", env1, start_key, sys_msg, temperature=temperature)

    response1 = process_code(response1)

    env2 = prompt2.replace("===qst===", data["question"]).replace("===tool===", response1)
    if task == "TabMWP":
        env2 = env2.replace("===table===", data["table"])

    response2 = gen_func("gpt-3.5-turbo-0613", env2, start_key, sys_msg, temperature=temperature)
    response2 = process_code(response2, is_api=True)

    print("-"*40)    

    ### retrieval ###
    if retrieval:
        response1 = "\n\n".join([toolbase["tool"][tool] for tool in tools]) + "\n" + "#"*40 + "\n" + response1         
        data.update({"retrieved_tools": tools})
    #################
    response = response1 + "\n" + "#"*40 + "\n" + response2

    data.update({"tool": response1, "tool_call": response2, "code": response})
    print("="*40)    

    return data


def create_shard(dataset, result_path, thread_id):
    dataset_dict = []

    if os.path.exists(result_path):
        annotated_dataset = json.load(open(result_path, "r"))
        dataset_dict.extend(annotated_dataset)
        if len(annotated_dataset) == len(dataset):
            return
        dataset = dataset.select(range(len(annotated_dataset), len(dataset)))

    for data in tqdm(dataset, total=len(dataset), desc=f"Inference: Thread {thread_id}"):
        dataset_dict.append(inference(data))
        # exit()
        lock.acquire()
        with open(result_path, "w") as f:
            json.dump([{k: v for k, v in data.items()} for data in dataset_dict], f)
        lock.release()
    
    lock.acquire()
    with open(result_path, "w") as f:
            json.dump([{k: v for k, v in data.items()} for data in dataset_dict], f)
    lock.release()
    return 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MATH", help="Choose from 'MATH', 'TabMWP', 'Creation'")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--thread", type=int, default=10)
    args = parser.parse_args()

    task = args.task
    retrieval = args.retrieval
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
    temperature = 0
    if retrieval:
        prompt_creation = f"{task}/prompt_lib/prompt_CREATOR_creation_with_retrieval.md"
    else:
        prompt_creation = f"{task}/prompt_lib/prompt_CREATOR_creation.md"
    prompt_decision = f"{task}/prompt_lib/prompt_CREATOR_decision.md"
    code_file = "code_exec/tmp0"
    gen_func = chat_api


    for field in fields[:1]:

        save_file = f"{task}/results/results_{field}_CREATOR.md" if not retrieval else \
                    f"{task}/results/results_{field}_CREATOR_retrieval.md"
        f = open(save_file, "w")
        f.close()
        
        f = open(prompt_creation, "r")
        prompt1 = f.read().strip()
        f.close
        
        f = open(prompt_decision, "r")
        prompt2 = f.read().strip()
        f.close

        dataset = datasets.Dataset.from_json(f"{task}/dataset/{field}.jsonl")
        if len(dataset) > 1000:
            if os.path.exists(f"{task}/dataset/{field}_1000.jsonl"):
                dataset = datasets.Dataset.from_json(f"{task}/dataset/{field}_1000.jsonl")
            else:
                dataset = dataset.shuffle().select(range(1000))
                dataset.to_json(f"{task}/dataset/{field}_1000.jsonl", orient="records", lines=True)

        if retrieval:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
            model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").cuda()
            
            toolbase = datasets.Dataset.from_csv(f'{task}/craft/{field}/5_deduplicated_tools.csv')
            vector_library = torch.load(f"{task}/craft/{field}/vector_library.pt")
        
        save_json_file = f"{task}/results/results_{field}_CREATOR.jsonl" if not retrieval \
                            else f"{task}/results/results_{field}_CREATOR_retrieval.jsonl"

        if os.path.exists(save_json_file):
            dataset = datasets.Dataset.from_json(save_json_file)
        else:
            lock = threading.Lock()
            threads = []
            num_threads = args.thread
            chunk_size = len(dataset) // num_threads
            idx = range(0, len(dataset), chunk_size)
            for i in range(num_threads):
                start = idx[i]
                end = idx[i+1] if i+1 < len(idx) else len(dataset)
                print(i, range(start, end))
                
                thread = threading.Thread(target=create_shard, args=(dataset.select(range(start, end)), 
                            save_json_file.replace("/results/", "/temp/").replace(".jsonl", f"_{i}.jsonl"), i))
                threads.append(thread)
            for thread in threads:
                thread.start()

            # Wait for each thread to finish
            for thread in threads:
                thread.join()

            # merge all jsonl
            dataset = datasets.concatenate_datasets([
                            datasets.Dataset.from_json(save_json_file.replace("/results/", "/temp/").replace(".jsonl", f"_{i}.jsonl")) 
                            for i in range(num_threads)])
            # save the whole dataset
            dataset.to_json(save_json_file, orient="records", lines=True)

        correct = 0
        wrong = 0
        exec_err = 0
        one_time_pass = 0
        use_tool = 0
        for data in iter(dataset):
            response1 = process_code(data["tool"])
            response2 = process_code(data["tool_call"], is_api=True)
            response = response1 + "\n" + "#"*40 + "\n" + response2
            correct_ans = data["answer"]
            time = 0
            print("="*40)
            try:
                time += 1
                if task == "TabMWP":
                    print("Table")
                    print(data["table"])
                print("Question:")
                print(data["question"])
                print("Response:")
                print(response)
                print("-"*40)
                if args.retrieval:
                    retrieved_tools = [toolbase["tool"][tool] for tool in data["retrieved_tools"]]
                    response1 = "\n\n".join(retrieved_tools) + "\n" + "#"*40 + "\n" + response1
                if_succ, info = execute_code(response1, "\n" + "#"*40 + "\n" + response2, code_file=code_file)
                if len(str(info).strip().split("\n")) > 1:
                    info = str(info).strip().split("\n")[-1]
                if if_succ:
                    print("~~~ Runing successfully ~~~")
                    correct_flag = False

                    if grade_answer(str(info), str(data["answer"])) or \
                        abs( (eval(str(info)) - eval(str(data["answer"]))) / eval(str(data["answer"])) ) < 0.01 or \
                        round(eval(str(info)), 2) == round(eval(str(data["answer"])), 2):

                        print("~~~ Correct Answer ~~~")
                        correct_flag = True
                        f = open(save_file, "a")
                        if task == "TabMWP":
                            f.write("### Table\n" + data["table"] + "\n")
                        f.write("### Question\n" + data["question"] + "\n### Respose\n" + response.strip() + "\n\n### Back info:\n" + info.strip() + f"\n\nCorrect Answer!\nThe correct answer should be {correct_ans}")
                        f.write("\n\n=============================split case=============================\n\n")
                        f.close()
                        correct += 1
                        if time == 1:
                            one_time_pass += 1
                                
                    if not correct_flag:
                        print("!!! Wrong Answer !!!")
                        f = open(save_file, "a")
                        if task == "TabMWP":
                            f.write("### Table\n" + data["table"] + "\n")
                        f.write("### Question\n" + data["question"] + "\n### Respose\n" + response.strip() + "\n\n### Back info:\n" + str(info).strip() + f"\n\nWrong Answer!\nThe correct answer should be {correct_ans}")
                        f.write("\n\n=============================split case=============================\n\n")
                        f.close()
                        wrong += 1
                else:
                    f = open(save_file, "a")
                    if task == "TabMWP":
                        f.write("### Table\n" + data["table"] + "\n")
                    f.write("### Question\n" + data["question"] + "\n### Respose\n" + response.strip() + "\n\n### Back info:\n" + str(info).strip() + f"\n\nExceed rectify limit!\nThe correct answer should be {correct_ans}")
                    f.write("\n\n=============================split case=============================\n\n")
                    f.close()
                    exec_err += 1
            except Exception as e: 
                print("Error in grader:")
                print("Info:", str(info), "Answer:", data["answer"])
                print(e)
                exec_err += 1    
            if retrieval:
                if len(data["retrieved_tools"] ) > 0:
                    tools = [extract_function_name(toolbase["tool"][tool]) for tool in data["retrieved_tools"]]
                    if any([tool in response1 for tool in tools]):
                        use_tool += 1
            print("prediction:", info, "answer:", correct_ans)
            print("Correct:", correct, "Wrong:", wrong, "Exec Error:", exec_err, "One Time Pass:", one_time_pass, "Tool Usage:", use_tool)
            print("="*40)
            print("\n"*2)
            


        f = open(save_file, "a")
        total_num = correct + wrong + exec_err
        f.write("Correct: " + str(correct) + " " + str(correct/total_num * 100) + "\nWrong: " + str(wrong) + " " + str(wrong/total_num * 100) + "\nExec Error: " + str(exec_err) + " " + str(exec_err/total_num * 100) + "\nOne Time Pass: " + str(one_time_pass) + "\nTotal: " + str(total_num) + "\n\n")
        f.close()
        print("Finish all!")