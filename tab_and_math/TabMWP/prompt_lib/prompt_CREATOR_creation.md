### Instruction
You are given a question with tabular contents, and you are asked to design python tools to help solve a question.
You can use pandas, numpy, math, etc. or other packages if necessary.
You are given a dataframe about a table. The first row is the name for each column. Each column is seperated by "|" and each row is seperated by "\n".
Pay attention to the format of the table, and what the question asks.
Generate the tools (functions) that may be used to solve the problem. Please wrap your codes in ```python ... ``` to make it one whole code block.

### Table
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
### Question
A candy dispenser put various numbers of orange candies into bags. How many bags had at least 32 orange candies?
### Solution
```python
import pandas as pd
def count_bags_with_32_orange_candies(df):
    """
    This function takes in a pandas dataframe of orange candies per bag, and returns the number of bags that have at least 32 orange candies.
    Args:
    df (pandas.DataFrame): A pandas DataFrame object containing the number of orange candies per bag.
    The dataframe should contain "Stem" and "Leaf" columns.
    Returns:
    int: The number of bags that have at least 32 orange candies.
    """
    # prepare a list to calculate candies in each bag
    candies = []
    # calculate the total number of orange candies in each bag
    for i in range(len(df)):
        stem = df['Stem'][i]
        leaf = df['Leaf'][i]
        for j in range(len(leaf)):
            candies.append(stem * 10 + leaf[j])
    # filter the bags where the total number of orange candies is greater than or equal to 32
    filtered = [candy for candy in candies if candy >= 32]
    # count the number of rows
    num_bags = len(filtered)
    return num_bags
```

### Table
pasta with meat sauce | $6.49
pasta with mushrooms | $9.05
spaghetti and meatballs | $7.43
mushroom pizza | $9.28
### Question
How much money does Jayla need to buy 5 orders of pasta with meat sauce and 3 orders of pasta with mushrooms?
### Solution
```python
import pandas as pd

def calculate_total_cost(menu_df, orders):
    """
    This function takes in a pandas DataFrame representing a menu table and a dictionary of orders, and returns the total cost of the orders using pandas.
    Args:
    menu_df (pd.DataFrame): A pandas DataFrame containing menu items and their prices with columns 'Item' and 'Price'.
    orders (dict): A dictionary where the keys are menu item names and the values are the number of orders for each item.
    Returns:
    float: The total cost of the orders.
    """
    # Initialize the total cost
    total_cost = 0.0
    
    # Iterate through the menu items and calculate the cost for each ordered item
    for item, quantity in orders.items():
        # Filter the DataFrame for the specific item
        item_df = menu_df[menu_df['Item'] == item]
        if not item_df.empty:
            item_price = item_df['Price'].values[0]
            total_cost += quantity * item_price
    
    return total_cost
```

### Table
===table===
### Question
===qst===
### Tools