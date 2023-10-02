### filter ###
filter_template = """You will be given a function named `llm_query`. The function Answers a text question using GPT-3 for reasoning and inference. Since GPT-3 cannot process visual information, the question must be image-independent.
Then, you will be given a query. You need to decide if this llm_query function is able to **directly** solve this query. Directly answer yes or no.
Tips: If the query requires visual information of an image to solve, you should answer no. Otherwise, if the query is an image-independent inference task that can be solved by LLM reasoning or a search engine, you should answer yes.

Query: Why isn't Song Hee taking customers?
Answer: yes

Query: Why might one of the news anchors look angry, and the other look concerned?
Answer: no

Query: {query}
Answer:"""



### eval ###
CONSISTENCY_PROMPT = """Given the question for the visual question answering task: {1}
Does the following predicted answer have the same meaning as the reference answer in the context of the question?
Predicted Answer: {2}
Reference Answer: {3}
You should compare the answers based on your understanding of the task, question, and answers, rather than relying on some superficial patterns like word overlap.
Directly answer Yes or No.
"""


### abstraction ###
abstraction_template = """**Rewriting Code to Create a Generic Tool Function**

**Purpose:** Given a query and its corresponding code solution, your task is to rewrite and abstract the code to create a general tool function that can be applied to similar queries. The tool function should solve a higher-level question that encompasses the original query and extend the code's functionality to make it more versatile and widely applicable.

Consider the following principles:

1. Understand the purpose of the query and infer a higher-level question that can be addressed by the tool function.
2. The generic tool function should solve queries of the same type, based on common reasoning steps rather than specific object types.
3. When enhancing the tool function's versatility according to the higher-level question, avoid assuming new attributes or methods of the `ImagePatch` classes.
4. Name the function honestly to ensure its functionality is not overclaimed. 
5. Avoid using redundant and unused variables or intermediate steps in the tool function. Ensure that every step in the code has a purpose and directly contributes to the desired outcome.
6. Replace specific strings or variable names with general variables to enhance the tool's applicability to various queries.
7. Provide a docstring and an example of how to call the tool function to solve a specific query.
8. End your answer with the format 'The final generic tool with docstring is: ...' and 'The example to call the tool is: ...'.
---

**Example**
Query: Is there a backpack to the right of the man?
Specific code solution: 
def execute_command(image)->str:
    image_patch = ImagePatch(image)
    man_patches = image_patch.find("man")
    if len(man_patches) == 0:
        # If no man is found, query the image directly with simple_query instead of returning a long string like "There is no man."
        return image_patch.simple_query("Is there a backpack to the right of the man?")
    man_patch = man_patches[0]
    backpack_patches = image_patch.find("backpack")
    if len(backpack_patches) == 0:
        return "no"
    for backpack_patch in backpack_patches:
        if backpack_patch.horizontal_center > man_patch.horizontal_center:
            return "yes"
    return "no"

Abstract tool:
The final generic tool with docstring is:
def check_existence_around_object_horizontally(image_patch: ImagePatch, object_name: str, reference_object_name: str, relative_horizontal_position: str, query: str) -> str:
    '''Check the existence of an object on either the left or right side of another object.
    
    Args:
        image_patch (ImagePatch): The image patch to check.
        object_name (str): The name of the object to check for existence.
        reference_object_name (str): The name of the reference object.
        relative_horizontal_position (str): The relative relative_horizontal_position position of the checked object to the reference object. Options: ["left", "right"].
        query (str): The original query to answer.
       
    Returns:
        str: "yes" if the object exists, "no" otherwise.
    '''
    
    assert relative_horizontal_position in ["left", "right"]
    reference_patches = image_patch.find(reference_object_name)
    if len(reference_patches) == 0:
        # If no reference object is found, query the image directly with simple_query instead of returning a long string like "There is no {reference_object_name}."
        return image_patch.simple_query(query)
    reference_patch = reference_patches[0]
    object_patches = image_patch.find(object_name)
    if len(object_patches) == 0:
        return "no"
    for object_patch in object_patches:
        if relative_horizontal_position == "left":
            flag = object_patch.horizontal_center < reference_patch.horizontal_center
        elif relative_horizontal_position == "right":
            flag = object_patch.horizontal_center > reference_patch.horizontal_center
        if flag:
            return "yes"
    return "no"

The example to call the tool is: check_existence_around_object_horizontally(image_patch, "backpack", "man", "right", "Is there a backpack to the right of the man?")


**Begin!**
Query: {query}
Specific code solution: 
{solution}

Abstract tool:
"""



# deduplication
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


# retrieval
retrieval_template = """Given a query, convert it into a declarative command and then a brief and concise imperative instruction. 
Next, infer tool functions that can be used based on the instruction. 
Finally, infer the docstring of the tool functions.

Consider the following principles:
1. The instruction should reflect the action to take, rather than emphasizing specific noun phrases. So you should prioritize using general terms like `object`, `people`, and `action`, and so on, instead of directly saying specific names like `desk`, `american president`, and `stuffed animal`.
2. Use tool function names following the format `verb_noun` with less than five words. Consider utilizing the most frequently used words in function names listed below.
3. The docstring of the tool function should be general and abstract, not specific to the query. Consider utilizing the most frequently used words in function docstrings listed below.
4. End your answer with the format 'The useful functions are: [...]' and 'The final answer is: ...', where '[...]' is a list of useful functions and '...' is the returned answer.
5. The most frequently used words in function names: ['object', 'identify', 'check', 'objects', 'find', 'attribute', 'action', 'location', 'determine', 'existence', 'infer', 'type', 'describe', 'property', 'image', 'purpose', 'activity', 'count', 'interaction', 'state']
6. The most frequently used words in function docstrings: ['specific', 'object', 'identify', 'check', 'image', 'certain', 'given', 'another', 'objects', 'find', 'type', 'existence', 'attribute', 'determine', 'action', 'possible', 'list', 'two', 'infer', 'number']

Query: What is visible on the desk?
Let's think step by step:
First, the corresponding declarative command of the query is 'Identify the visible objects on the desk'. After abstracting, the general instruction should be 'Identify the objects on the specific surface.'.
So considering the naming rules of tool functions, the relevant and useful functions could be named as 'identify_objects' or 'identify_objects_on_surface'.
Finally, we can infer that the docstring of the tool function could be 'Identify the objects on the specified surface.'.
The useful functions are: ['identify_objects', 'identify_objects_on_surface'].
The final answer is: Identify the objects on the specified surface.

Query: Which american president is most associated with the stuffed animal seen here?
Let's think step by step:
First, the corresponding declaritive command of the query is 'Search the american president most associated with the stuffed animal seen here'.\n\n"\
After abstracting, the general instruction should be 'Search people most associated with the specific object.'.\n\n"\
So considering the naming rules of tool functions, the relevant and useful functions could be named as 'search_people_associated_with_object'.\n\n"\
Finally, we can infer that the docstring of the tool function could be 'Search for people most associated with the specific object.'.\n\n"\
The useful functions are: ['search_people_associated_with_object'].\n\n"\
The final answer is: Search for people most associated with a specific object.

Query: {query}
Let's think step by step:
"""


