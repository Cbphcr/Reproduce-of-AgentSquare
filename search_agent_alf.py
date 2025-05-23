import time
import json
import openai
import traceback
from copy import deepcopy
from agent import Agent
from alfworld_run import test_alfworld
from module.prompt_reasoning import get_init_archive_reasoning, get_prompt_reasoning
from module.prompt_planning import get_init_archive_planning, get_prompt_planning
from module.prompt_memory import get_init_archive_memory, get_prompt_memory
from planning_modules import PlanningBase
from reasoning_modules import ReasoningBase
from memory_modules import MemoryBase

import os
import re
import ast
import uuid
import shutil
from utils import llm_response
from collections import Counter
from planning_prompt import planning_prompt
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

task_description = "ALFworld is a suite of text-based environments that challenge an agent to solve multi-step tasks in a variety of interactive environments. It includes 6 types of tasks in which an agent needs to achieve a high-level goal (e.g. examine paper under desklamp) by navigating and interacting with a simulated household via text actions (e.g. go to coffeetable 1, take paper 2, use desklamp 1)."


def get_json_response(msg_list, model="gpt-3.5-turbo-instruct", temperature=0.8):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature,
        stop=None,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content) if content else None
    assert not json_dict is None
    return json_dict


class AgentSearcher:
    def __init__(self, verbose=True, logger_file_path="./log.txt"):
        """
        Initialize the AgentSearcher with the given environment.
        :param env: The environment in which the agent will operate.
        """
        # Initialize the environment and the archives for reasoning, planning, memory, and tool use.
        self.verbose = verbose
        self.exec_globals = {
            "os": os,
            "re": re,
            "ast": ast,
            "uuid": uuid,
            "shutil": shutil,
            "Chroma": Chroma,
            "Counter": Counter,
            "Document": Document,
            "MemoryBase": MemoryBase,
            "llm_response": llm_response,
            "PlanningBase": PlanningBase,
            "ReasoningBase": ReasoningBase,
            "planning_prompt": planning_prompt,
            "OpenAIEmbeddings": OpenAIEmbeddings,
        }

        self.archive_reasoning_pool = get_init_archive_reasoning()
        self.archive_planning_pool = get_init_archive_planning()
        self.archive_memory_pool = get_init_archive_memory()
        # Initialize the agent archive with random selections from the pools.
        # The agent archive contains the reasoning, planning, memory, and tool use modules.

        if os.path.exists(logger_file_path):
            # raise ValueError(
            #     f"Logger file {logger_file_path} already exists. Please remove it or choose a different name."
            # )
            os.remove(logger_file_path)
        self.logger_file_path = logger_file_path
        with open(self.logger_file_path, "w") as f:
            f.write("Agent Searcher Log\n")
            f.write("=" * 50 + "\n")
            f.write("Timestamp: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("=" * 50 + "\n")
            f.write("Initialized AgentSearcher with the following parameters:\n")
            f.write("Verbose: {}\n".format(self.verbose))
            f.write("Logger file path: {}\n".format(self.logger_file_path))

        # Use `llm_generated_modules` dir to store the generated modules.
        if not os.path.exists("llm_generated_modules"):
            os.makedirs("llm_generated_modules")
        if not os.path.exists("llm_generated_modules/planning"):
            os.makedirs("llm_generated_modules/planning")
        if not os.path.exists("llm_generated_modules/reasoning"):
            os.makedirs("llm_generated_modules/reasoning")
        if not os.path.exists("llm_generated_modules/memory"):
            os.makedirs("llm_generated_modules/memory")
        self.test_size = 0
        self.max_episodes = 0
        self.population_size = 0
        self.tested_cases = []
        # self.cur_agent_archive = {
        #     "planning": "io",
        #     "reasoning": "io",
        #     "tooluse": None,
        #     "memory": "tp",
        #     "performance": 0,
        # }

    def _get_module_from_string(self, module_str):
        """
        Convert a string representation of a module to an actual module.
        :param module_str: The string representation of the module.
        :return: The actual module.
        """
        local_vars = {}
        try:
            exec(module_str, self.exec_globals, local_vars)
            return next(v for v in local_vars.values() if isinstance(v, type))
        except Exception as e:
            print(f"Error in _get_module_from_string: {e}")
            traceback.print_exc()
            with open(self.logger_file_path, "a") as f:
                f.write("Error in _get_module_from_string: {}\n".format(e))
                f.write("Module string: {}\n".format(module_str))
                f.write("=" * 50 + "\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("=" * 50 + "\n")
            exit(1)
        return None

    def _convert_archive_to_agent(self, agent_archive, new_code="", new_module_type=""):
        """
        Convert an agent archive to an actual agent.
        :param agent_archive: The agent archive.
        :return: The actual agent.
        """
        print("archive:", agent_archive)
        reasoning_cls = None
        planning_cls = None
        memory_cls = None
        tooluse_cls = None
        for reasoning_module in self.archive_reasoning_pool:
            if reasoning_module["name"] == agent_archive["reasoning"]:
                reasoning_cls = self._get_module_from_string(reasoning_module["code"])
                break
        for planning_module in self.archive_planning_pool:
            if planning_module["name"] == agent_archive["planning"]:
                planning_cls = self._get_module_from_string(planning_module["code"])
                break
        for memory_module in self.archive_memory_pool:
            if memory_module["name"] == agent_archive["memory"]:
                memory_cls = self._get_module_from_string(memory_module["code"])
                break

        if new_module_type == "reasoning":
            reasoning_cls = self._get_module_from_string(new_code)
        elif new_module_type == "planning":
            planning_cls = self._get_module_from_string(new_code)
        elif new_module_type == "memory":
            memory_cls = self._get_module_from_string(new_code)

        if not (reasoning_cls and planning_cls and memory_cls):
            print(
                "Error: Failed to convert archive to agent. Missing modules in the archive."
            )
            if reasoning_cls is None:
                print("Reasoning module not found in the archive.")
            if planning_cls is None:
                print("Planning module not found in the archive.")
            if memory_cls is None:
                print("Memory module not found in the archive.")
            return None, False
        else:
            return (
                Agent(
                    name="alfworld_agent",
                    profile="",
                    MEMORY=memory_cls,
                    REASONING=reasoning_cls,
                    PLANNING=planning_cls,
                    TOOLUSE=tooluse_cls,
                ),
                True,
            )

    def run(self, test_size=10, max_episodes=3, population_size=3):
        """
        Run the agent searcher with the given parameters.
        """
        self.test_size = test_size
        self.max_episodes = max_episodes
        self.population_size = population_size
        self.cur_agent_archive = {
            "planning": "io",
            "reasoning": "io",
            "tooluse": None,
            "memory": "tp",
            "performance": 0,
        }
        # self.cur_agent_archive["performance"] = self._test(
        #     self._convert_archive_to_agent(self.cur_agent_archive)[0]
        # )
        agent, success = self._convert_archive_to_agent(self.cur_agent_archive)
        if not success:
            print("Error: Failed to convert default archive to agent.")
            return None
        self.cur_agent_archive["performance"] = self._test(agent)
        self.tested_cases.append(self.cur_agent_archive)
        for episode in range(self.max_episodes):
            print(f"Episode {episode + 1}/{self.max_episodes}")

            # 1. Evolution of the agent.
            # Evolve the agent by generating new solutions based on the current archive.
            self._evolution()

            # 2. Recombination of the agent.
            # Recombine the current archive to create new solutions.
            self._recombination()

        print("Final Best Agent Archive:")
        best_agent = {"performance": 0}
        for agent in self.tested_cases:
            if best_agent is None or agent["performance"] > best_agent["performance"]:
                best_agent = agent
        print(best_agent)
        with open("agent_archive.json", "w") as f:
            json.dump(self.tested_cases, f, indent=4)
        with open("best_agent.json", "w") as f:
            json.dump(best_agent, f, indent=4)
        return best_agent, self.tested_cases

    def _evolution(self):
        print("Evolving new agents...")
        tmp_best_agent_archive = deepcopy(self.cur_agent_archive)
        # 1. Generate new modules
        system_prompt, prompt = get_prompt_reasoning(self.archive_reasoning_pool)
        msg_list_reasoning = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        system_prompt, prompt = get_prompt_planning(self.archive_planning_pool)
        msg_list_planning = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        system_prompt, prompt = get_prompt_memory(self.archive_memory_pool)
        msg_list_memory = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # 2.1 evaluate the new reasoning module
        next_solution_reasoning = get_json_response(msg_list_reasoning)
        print("Successfully generate reasoning module.")
        agent_archive_with_new_reasoning = deepcopy(self.cur_agent_archive)
        agent_archive_with_new_reasoning["reasoning"] = next_solution_reasoning["name"]
        agent_with_new_reasoning, success = self._convert_archive_to_agent(
            agent_archive_with_new_reasoning,
            new_code=next_solution_reasoning["code"],
            new_module_type="reasoning",
        )
        if success:
            try:
                print("Testing the new reasoning module...")
                performance = self._test(agent_with_new_reasoning)
                del agent_with_new_reasoning
                next_solution_reasoning["performance"] = performance
                agent_archive_with_new_reasoning["performance"] = performance
                self.tested_cases.append(agent_archive_with_new_reasoning)
                if performance > tmp_best_agent_archive["performance"]:
                    tmp_best_agent_archive = deepcopy(agent_archive_with_new_reasoning)
                self.archive_reasoning_pool.append(next_solution_reasoning)
                with open(
                    "llm_generated_modules/reasoning/{}.json".format(
                        next_solution_reasoning["name"]
                    ),
                    "w",
                ) as f:
                    json.dump(next_solution_reasoning, f, indent=4)
            except Exception as e:
                print("Error in testing the new reasoning module:", e)
                with open(self.logger_file_path, "a") as f:
                    f.write("Error in testing the new reasoning module: {}\n".format(e))
                    f.write("Module: {}\n".format(next_solution_reasoning))
                    f.write("=" * 50 + "\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("=" * 50 + "\n")
        else:
            print(
                "Error: Failed to convert archive to agent with new reasoning module."
            )
            with open(self.logger_file_path, "a") as f:
                f.write(
                    "Error: Failed to convert archive to agent with new reasoning module.\n"
                )
                f.write("Module: {}\n".format(next_solution_reasoning))
                f.write("=" * 50 + "\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("=" * 50 + "\n")

        # 2.2 evaluate the new planning module
        next_solution_planning = get_json_response(msg_list_planning)
        print("Successfully generate planning module.")
        agent_archive_with_new_planning = deepcopy(self.cur_agent_archive)
        agent_archive_with_new_planning["planning"] = next_solution_planning["name"]
        agent_with_new_planning, success = self._convert_archive_to_agent(
            agent_archive_with_new_planning,
            new_code=next_solution_planning["code"],
            new_module_type="planning",
        )
        if success:
            try:
                print("Testing the new planning module...")
                performance = self._test(agent_with_new_planning)
                del agent_with_new_planning
                next_solution_planning["performance"] = performance
                agent_archive_with_new_planning["performance"] = performance
                self.tested_cases.append(agent_archive_with_new_planning)
                if performance > tmp_best_agent_archive["performance"]:
                    tmp_best_agent_archive = deepcopy(agent_archive_with_new_planning)
                self.archive_planning_pool.append(next_solution_planning)
                with open(
                    "llm_generated_modules/planning/{}.json".format(
                        next_solution_planning["name"]
                    ),
                    "w",
                ) as f:
                    json.dump(next_solution_planning, f, indent=4)
            except Exception as e:
                print("Error in testing the new planning module:", e)
                with open(self.logger_file_path, "a") as f:
                    f.write("Error in testing the new planning module: {}\n".format(e))
                    f.write("Module: {}\n".format(next_solution_planning))
                    f.write("=" * 50 + "\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("=" * 50 + "\n")
        else:
            print("Error: Failed to convert archive to agent with new planning module.")
            with open(self.logger_file_path, "a") as f:
                f.write(
                    "Error: Failed to convert archive to agent with new planning module.\n"
                )
                f.write("Module: {}\n".format(next_solution_planning))
                f.write("=" * 50 + "\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("=" * 50 + "\n")
        # 2.3 evaluate the new memory module
        next_solution_memory = get_json_response(msg_list_memory)
        print("Successfully generate memory module.")
        agent_archive_with_new_memory = deepcopy(self.cur_agent_archive)
        agent_archive_with_new_memory["memory"] = next_solution_memory["name"]
        agent_with_new_memory, success = self._convert_archive_to_agent(
            agent_archive_with_new_memory,
            new_code=next_solution_memory["code"],
            new_module_type="memory",
        )
        if success:
            try:
                print("Testing the new memory module...")
                performance = self._test(agent_with_new_memory)
                del agent_with_new_memory
                next_solution_memory["performance"] = performance
                agent_archive_with_new_memory["performance"] = performance
                self.tested_cases.append(agent_archive_with_new_memory)
                if performance > tmp_best_agent_archive["performance"]:
                    tmp_best_agent_archive = deepcopy(agent_archive_with_new_memory)
                self.archive_memory_pool.append(next_solution_memory)
                with open(
                    "llm_generated_modules/memory/{}.json".format(
                        next_solution_memory["name"]
                    ),
                    "w",
                ) as f:
                    json.dump(next_solution_memory, f, indent=4)
            except Exception as e:
                print("Error in testing the new memory module:", e)
                with open(self.logger_file_path, "a") as f:
                    f.write("Error in testing the new memory module: {}\n".format(e))
                    f.write("Module: {}\n".format(next_solution_memory))
                    f.write("=" * 50 + "\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("=" * 50 + "\n")
        else:
            print("Error: Failed to convert archive to agent with new memory module.")
            with open(self.logger_file_path, "a") as f:
                f.write(
                    "Error: Failed to convert archive to agent with new memory module.\n"
                )
                f.write("Module: {}\n".format(next_solution_memory))
                f.write("=" * 50 + "\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("=" * 50 + "\n")

        # 3. Update the current best agent archive
        self.cur_agent_archive = tmp_best_agent_archive

    def _recombination(self):
        for i in range(self.population_size):
            tooluse_candidate = {"None": "ToolUse is not needed for this task."}
            planning_candidate = {
                module["name"]: module["thought"]
                for module in self.archive_planning_pool
            }
            reasoning_candidate = {
                module["name"]: module["thought"]
                for module in self.archive_reasoning_pool
            }
            memory_candidate = {
                module["name"]: module["thought"] for module in self.archive_memory_pool
            }
            prompt = (
                "You are an AI agent expert. Now you are required to design a LLM-based agent to solve the task of "
                + task_description
                + "The agent is composed of four fundamental modules: planning, reasoning, tool use and memory. \
            For each module you are required to choose one from the follwing provided candidates. \
            Planning module candidates and descriptions: "
                + str(planning_candidate)
                + " Reasoning module candidates and descriptions: "
                + str(reasoning_candidate)
                + " Tool use module candidates and descriptions: "
                + str(tooluse_candidate)
                + " Memory module candidates and descriptions: "
                + str(memory_candidate)
                + "The performance of some existing module combinations: "
                + str(self.tested_cases)
                + ". "
                + "You are expected to give a new module combination to improve the performance on the task by considering (1) the matching degree between the module description and task description (2) performance of existing module combinations on the task. \
            Your answer should follow the json format:"
                + str(
                    {
                        "planning": "<your choice>",
                        "reasoning": "<your choice>",
                        "tooluse": "<your choice>",
                        "memory": "<your choice>",
                    }
                )
            )
            response = get_json_response(
                [
                    {"role": "system", "content": "You are an AI agent expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            new_agent_archive = response
            new_agent_archive["predict_performance"] = self._predictor(
                new_agent_archive
            )
            try:
                agent, success = self._convert_archive_to_agent(new_agent_archive)
                if not success:
                    print(
                        "Error: Failed to convert archive to agent with new module combination."
                    )
                    continue
                new_agent_archive["performance"] = self._test(agent)
                self.tested_cases.append(new_agent_archive)
                if (
                    new_agent_archive["performance"]
                    > self.cur_agent_archive["performance"]
                ):
                    self.cur_agent_archive = new_agent_archive
                    print("New best agent archive found:", self.cur_agent_archive)
            except Exception as e:
                print("Error in testing the new module combination:", e)
                with open(self.logger_file_path, "a") as f:
                    f.write(
                        "Error in testing the new module combination: {}\n".format(e)
                    )
                    f.write("Module: {}\n".format(new_agent_archive))
                    f.write("=" * 50 + "\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("=" * 50 + "\n")
            print("Recombination response:", response)

    def _test(self, agent):
        with open(self.logger_file_path, "a") as f:
            f.write("Testing agent: {}\n".format(agent))
            f.write("Planning module: {}\n".format(agent.planning))
            f.write("Reasoning module: {}\n".format(agent.reasoning))
            f.write("Memory module: {}\n".format(agent.memory))
            f.write("Tool use module: {}\n".format(agent.tooluse))
            f.write("=" * 50 + "\n")
        return test_alfworld(
            agent=agent,
            max_episodes=self.test_size,
        )

    def _predictor(self, agent_archive):
        tooluse_candidate = {"None": "ToolUse is not needed for this task."}
        planning_candidate = {
            module["name"]: module["thought"] for module in self.archive_planning_pool
        }
        reasoning_candidate = {
            module["name"]: module["thought"] for module in self.archive_reasoning_pool
        }
        memory_candidate = {
            module["name"]: module["thought"] for module in self.archive_memory_pool
        }
        tested_case = self.tested_cases
        agent_archive["performance"] = 0
        predict_case = agent_archive
        prompt = f"""You are a performance estimator. Now you need to estimate the performance of a LLM-based agent solving an ALFworld task (sequential decision making tasks with steps including finding hidden objects, moving objects and manipulating objects with other objects ). The agent is composed of four fundamental modules: planning, reasoning, tool use and memory. Planning module candidates and descriptions: {planning_candidate}, Reasoning module candidates and descriptions: {reasoning_candidate}, Tool use module candidates and descriptions: {tooluse_candidate}, Memory module candidates and descriptions: {memory_candidate}. The performance of some existing module combinations: {tested_case}. The module combination you need to predict: {predict_case}. Be sure to give exact predictions in 0-1 and respond in json format {{"performance": <your choice>}}"""
        response = get_json_response(
            [
                {"role": "system", "content": "You are a performance estimator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        print("Predictor response:", response)
        if response.get("performance") is not None:
            return response["performance"]
        else:
            raise ValueError("Invalid response from predictor: {}".format(response))
        return 0


if __name__ == "__main__":
    env = "ALFWorld"
    agent_searcher = AgentSearcher()
    # agent_searcher.run(test_size=1, max_episodes=1, population_size=1)
    agent_searcher.run(test_size=10, max_episodes=3, population_size=4)
