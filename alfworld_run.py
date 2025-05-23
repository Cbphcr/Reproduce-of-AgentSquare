import os
import re
import logging
from utils import get_price
from workflow import workflow
import yaml
import alfworld
import alfworld.agents.environment
import argparse

with open("base_config.yaml") as reader:
    config = yaml.safe_load(reader)

split = "eval_out_of_distribution"


import json

folder = "./prompts/"
prompt_file = "alfworld_3prompts.json"
with open(folder + prompt_file, "r") as f:
    d = json.load(f)
folder = "./prompts/"
prompt_file = "alfworld_reasoning_prompt1.json"
with open(folder + prompt_file, "r") as f:
    exp = json.load(f)

import sys


def alfworld_run(env, prompt_exp, alfworld_solver, v, reason_exp, to_print=True, ob=""):
    class ALFWORLD:
        def __init__(
            self, env, prompt_exp, alfworld_solver, v, reason_exp, to_print=True, ob=""
        ):
            self.init_prompt = prompt_exp + ob + "\n>"
            prompt = ""
            if to_print:
                print(ob)
                sys.stdout.flush()
            if alfworld_solver.planning is None:
                self.task_description = self.init_prompt + prompt
            else:
                self.task_description = ob
            self.prompt = prompt
            self.ob = ob
            self.task_type = v
            self.reason_exp = reason_exp
            # self.init_prompt = init_prompt
            self.last_action = ""
            self.memory_pool = []
            self.env = env
            self.max_step_number = 50
            self.max_step_number_plan = 20
            self.tool_instruction = ""
            self.feedback_previous_tools = ""
            print("Start with Observation: ", ob)

        def step(self, action):
            action[0] = (
                action[0].replace(">", "").replace("OK.", "").replace("OK", "").strip()
            )
            observation, reward, done, info = self.env.step(
                [
                    action[0]
                    .replace(">", "")
                    .replace("OK.", "")
                    .replace("OK", "")
                    .strip()
                    .split(", end")[0]
                    .replace(".", "")
                ]
            )

            def process_ob(ob):
                if ob.startswith("You arrive at loc "):
                    ob = ob[ob.find(". ") + 2 :]
                return ob

            # 处理 observation、reward 和 done
            observation = process_ob(observation[0])

            processed_reward = info["won"][0]
            processed_done = done[0]
            if action[0].startswith("think:"):
                observation = "OK."
            self.prompt += f" {action[0]}\n{observation}\n>"
            self.task_description = self.init_prompt + self.prompt
            # 返回处理后的值
            return observation, processed_reward, processed_done

        def prompt_exp_update(self, sub_task_id):
            values = list(self.reason_exp.values())
            prompt_exp = values[sub_task_id]
            return prompt_exp

        def prompt_reset(self):
            self.prompt = ""

        def memory_update(self):
            return (self.ob + "\n>" + self.prompt)[:-1] + "success."

        def memory_cache(self, sub_tasks, sub_task_id):
            self.memory_pool.append(
                (
                    self.ob.split(":")[0]
                    + ": "
                    + "Last step: "
                    + self.last_action
                    + ". "
                    + "Current task: "
                    + sub_tasks[sub_task_id]["reasoning instruction"]
                    + ".\n>"
                    + self.prompt
                )[:-1]
                + "success."
            )
            # return self.memory_pool

        def init_prompt_update(self, sub_tasks, sub_task_id):
            return (
                self.ob.split(":")[0]
                + ": "
                + "Last step: "
                + self.last_action
                + ". "
                + "Current task: "
                + sub_tasks[sub_task_id]["reasoning instruction"]
                + ".\n>"
            )

        def flag(self, action, sub_tasks, sub_task_id):
            if "end" in action or " complete" in action:
                self.last_action = action
                return True
            else:
                return False

    # 使用封装的 CustomEnv
    alfworld = ALFWORLD(
        env, prompt_exp, alfworld_solver, v, reason_exp, to_print=True, ob=ob
    )

    return workflow(alfworld_solver, alfworld)


def run_episodes(alfworld_solver, log_path=None):
    # 设置日志路径
    if log_path is None:
        script_path = os.path.abspath(__file__)
        log_path = os.path.splitext(script_path)[0] + ".log"

    # 设置前缀映射
    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }

    cnts = [0] * 6
    rs = [0] * 6

    # 检查日志文件中的历史记录，获取已完成的 episode 数量
    completed = 0
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                match = re.match(r"^(\d+) r", line)
                if match:
                    completed = max(completed, int(match.group(1)))
                if "rs [" in line and "cnts [" in line:
                    try:
                        rs_str = re.search(r"rs \[([^\]]+)\]", line).group(1)
                        cnts_str = re.search(r"cnts \[([^\]]+)\]", line).group(1)
                        rs = list(map(int, rs_str.strip().split(",")))
                        cnts = list(map(int, cnts_str.strip().split(",")))
                    except Exception as e:
                        print("Failed to parse rs and cnts from log:", e)
                if (
                    "completion_tokens" in line
                    and "prompt_tokens" in line
                    and "price=" in line
                ):
                    try:
                        comp = re.search(r"completion_tokens:(\d+)", line)
                        promp = re.search(r"prompt_tokens:(\d+)", line)
                        completion_tokens = int(comp.group(1)) if comp else 0
                        prompt_tokens = int(promp.group(1)) if promp else 0

                    except Exception as e:
                        print("Failed to parse token info from log:", e)

    # 自定义日志函数，同时输出到控制台和文件
    def log_info(message):
        print(message)
        with open(log_path, "a") as f:
            f.write(message + "\n")

    log_info(f"Resuming from episode {completed+1}")

    for ep in range(0, 134):
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
        log_info(name)
        if ep < completed:
            print(f"Skipping completed episode {ep+1}: {name}")
            continue
        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                prompt = d[f"react_{v}_2"] + d[f"react_{v}_0"] + "\nHere is the task.\n"
                log_info(f"{k} {v}")
                reason_exp = exp[v]
                r = alfworld_run(env, prompt, alfworld_solver, v, reason_exp, ob=ob)
                rs[i] += r
                cnts[i] += 1
                break

        result_str = (
            f"{ep+1} r {r} rs {rs} cnts {cnts} sum(rs)/sum(cnts) {sum(rs) / sum(cnts)}"
        )
        log_info(result_str)

        completion_tokens, prompt_tokens, _ = get_price()
        log_info(
            f"completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price={completion_tokens*15/1e6+prompt_tokens*5/1e6}"
        )
        log_info("------------\n")

    return sum(rs) / sum(cnts)


from agent import Agent
from module_map import ModuleMap


def run_alfworld(
    planning=None,
    reasoning=None,
    tooluse=None,
    memory=None,
    llms_type=["gpt=3.5-turbo-0125"],
):
    planning_module, reasoning_module, tooluse_module, memory_module = ModuleMap(
        planning, reasoning, tooluse, memory
    )
    alfworld_solver = Agent(
        "alfworld_solver",
        "",
        memory_module,
        reasoning_module,
        tooluse_module,
        planning_module,
        llms_type,
    )
    res1 = run_episodes(alfworld_solver)
    return res1


def test_alfworld(agent: Agent, max_episodes=25):
    env = getattr(alfworld.agents.environment, config["env"]["type"])(
        config, train_eval=split
    )
    env = env.init_env(batch_size=1)
    alfworld_solver = agent
    # 设置前缀映射
    prefixes = {
        "pick_and_place": "put",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "look_at_obj": "examine",
        "pick_two_obj": "puttwo",
    }

    cnts = [0] * 6
    rs = [0] * 6

    # for ep in range(0, 134):
    print(min(134, max_episodes))
    for ep in range(0, min(134, max_episodes)):
        print(f"Episode in ALFWorld {ep+1}/{min(134, max_episodes)}")
        ob, info = env.reset()
        ob = "\n".join(ob[0].split("\n\n")[1:])
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        for i, (k, v) in enumerate(prefixes.items()):
            if name.startswith(k):
                prompt = d[f"react_{v}_2"] + d[f"react_{v}_0"] + "\nHere is the task.\n"
                reason_exp = exp[v]
                r = alfworld_run(env, prompt, alfworld_solver, v, reason_exp, ob=ob)
                rs[i] += r
                cnts[i] += 1
                print(
                    f"r {r} rs {rs} cnts {cnts} sum(rs)/sum(cnts) {sum(rs) / sum(cnts)}"
                )
                break

        # result_str = (
        #     f"{ep+1} r {r} rs {rs} cnts {cnts} sum(rs)/sum(cnts) {sum(rs) / sum(cnts)}"
        # )
        # completion_tokens, prompt_tokens, _ = get_price()
    print(f"Final result: rs {rs} cnts {cnts} sum(rs)/sum(cnts) {sum(rs) / sum(cnts)}")
    return sum(rs) / sum(cnts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ALFWorld with specified modules.")
    parser.add_argument(
        "--planning", type=str, default="none", help="Specify planning module"
    )
    parser.add_argument(
        "--reasoning", type=str, default="io", help="Specify reasoning module"
    )
    parser.add_argument(
        "--tooluse",
        type=str,
        default="none",
        help="tooluse is not required in ALFworld",
    )
    parser.add_argument(
        "--memory", type=str, default="none", help="Specify memory module"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-instruct",
        help="Specify the LLM model type",
    )
    args = parser.parse_args()
    run_alfworld(
        planning=args.planning,
        reasoning=args.reasoning,
        tooluse=args.tooluse,
        memory=args.memory,
        llms_type=[args.model],
    )
