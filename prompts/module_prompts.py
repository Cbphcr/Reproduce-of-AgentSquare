alf_task_desc = """ALFworld is a suite of text-based environments that challenge an agent to solve multi-step tasks in a variety of interactive environments. It includes 6 types of tasks in which an agent needs to achieve a high-level goal (e.g. examine paper under desklamp) by navigating and interacting with a simulated household via text actions (e.g. go to coffeetable 1, take paper 2, use desklamp 1)."""

def get_recombination_prompt(task_description, planning_candidate, reasoning_candidate, tooluse_candidate, memory_candidate, tested_case):
    """
    Generate a prompt for module recombination.
    Args:
        task_description (str): Description of the task.
        planning_candidate (list): List of planning module candidates and their descriptions.
        reasoning_candidate (list): List of reasoning module candidates and their descriptions.
        tooluse_candidate (list): List of tool use module candidates and their descriptions.
        memory_candidate (list): List of memory module candidates and their descriptions.
        tested_case (list): List of existing module combinations and their performance on the task.
    Returns:
        str: The generated prompt.
    """
    prompt = 'You are an AI agent expert. Now you are required to design a LLM-based agent to solve the task of ' \
    + task_description + 'The agent is composed of four fundamental modules: planning, reasoning, tool use and memory. \
    For each module you are required to choose one from the follwing provided candidates. \
    Planning module candidates and descriptions: ' + str(planning_candidate) + ' Reasoning module candidates and descriptions: ' + str(reasoning_candidate) + ' Tool use module candidates and descriptions: ' + str(tooluse_candidate) + ' Memory module candidates and descriptions: ' + str(memory_candidate) \
    + 'The performance of some existing module combinations: ' + str(tested_case) +'. ' \
    + 'You are expected to give a new module combination to improve the performance on the task by considering (1) the matching degree between the module description and task description (2) performance of existing module combinations on the task. \
    Your answer should follow the format:' +str({'planning': '<your choice>', 'reasoning': '<your choice>', 'tooluse': '<your choice>', 'memory': '<your choice>'})
    return prompt