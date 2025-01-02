import numpy as np
import json
import jsonlines
import random
from mat.utils.language_buffer import LanguageBuffer
from copy import deepcopy
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE
import numpy as np
# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_path, mode):
    if "jsonl" in dataset_path:
        with jsonlines.open(dataset_path) as reader:
            dataset = [line for line in reader]
    else:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    return dataset

class MathEnv:

    def __init__(self, rank, dataset_name, dataset_path, mode, max_steps):
        
        self.rank = rank
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_path=dataset_path, mode=mode)
        self.n_agents = 1
        self.max_step = max_steps
        self.step_count = 0
        
        if self.mode == "test":
            self.problem_idx = 0
        
        self.problem = None
        self.label = None
        self.step_tag = "ки"
        self.current_state = None
        self.trajectory = None

    def reset(self):
        
        problem_answer_pair = random.choice(self.dataset)
        # problem_answer_pair = self.dataset[3]
        # if self.mode == "test":
        #     problem_answer_pair = self.dataset[self.problem_idx]
        #     self.problem_idx  = (self.problem_idx + 1) % len(self.dataset)
        print("RESET CALLED")    
        self.problem = problem_answer_pair["problem"]
        self.label = problem_answer_pair["final_answer"]
        
        print(f"\n\n\n\n======== new problem: {self.problem}, label: {self.label} ==========", )
        
        self.current_state = IN_CONTEXT_EXAMPLE + self.problem + "\n"
        # we can make trajectory a
        obs = np.array([self.current_state], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, action):
        self.step_count += 1
        # insert something here to have a version where we don't replace the self.step_tag, for AI prm labelling at the end
        step_tag_action = action[0]
        action = action[0]
        action = action.replace(self.step_tag, "").strip()
        # they take out action steps here -- interesting
        print(f"action: {action}")
        self.current_state = self.current_state + action + " " + self.step_tag + "\n" # so this removes the step tags and preserves it, we want to keep using this to get the actions

        # self.current_state = self.current_state + action.strip() + "\n"
        next_obs = np.array([self.current_state], dtype=np.object_)
        
        score = 0.0
        if "step" in action.lower() or "answer" in action.lower():
            score = 1.0
        if "answer" in action.lower():
            dones = np.ones((self.n_agents), dtype=bool)
        elif self.step_count >= self.max_step:
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
        
        rewards = [score for _ in range(self.n_agents)]
        infos = {"state": self.current_state}
        return next_obs, rewards, dones, infos

    def seed(self, seed):
        np.random.seed(seed)


class TrajectoryMathEnv(MathEnv):

    def __init__(self, rank, dataset_name, dataset_path, mode, max_steps, max_new_tokens):
        super().__init__(rank, dataset_name, dataset_path, mode, max_steps)
        # n_agents is going to be 1 for now and we just increase vectorized env
        self.max_new_tokens = max_new_tokens
        self.obs = np.empty((max_steps + 1, self.n_agents), dtype=np.object_)
        self.actions = np.empty((max_steps, self.n_agents), dtype=np.object_)
        self.action_tokens = np.empty((max_steps, self.n_agents, self.max_new_tokens), dtype=np.int64)
        self.log_probs = np.empty((max_steps, self.n_agents), dtype=np.float32)
        self.values = np.empty((max_steps, self.n_agents), dtype=np.float32)
        self.reasoning_chain = ""

    def reset(self):
        
        problem_answer_pair = random.choice(self.dataset)
        # problem_answer_pair = self.dataset[3]
        # if self.mode == "test":
        #     problem_answer_pair = self.dataset[self.problem_idx]
        #     self.problem_idx  = (self.problem_idx + 1) % len(self.dataset) 
        self.problem = problem_answer_pair["problem"]
        self.label = problem_answer_pair["final_answer"]
        
        print(f"\n\n\n\n======== new problem: {self.problem}, label: {self.label} ==========", )
        self.current_state = IN_CONTEXT_EXAMPLE + self.problem + "\n"
        self.reasoning_chain = self.problem + "\n"
        obs = np.array([self.current_state], dtype=np.object_)
        self.obs = np.empty((self.max_step + 1, self.n_agents), dtype=np.object_)
        self.obs.fill(-100)
        self.actions = np.empty((self.max_step, self.n_agents), dtype=np.object_)
        self.actions.fill(-100)
        self.action_tokens = np.empty((self.max_step, self.n_agents, self.max_new_tokens), dtype=np.int64)
        self.log_probs = np.empty((self.max_step, self.n_agents), dtype=np.float32)
        self.log_probs.fill(-100)
        self.values = np.empty((self.max_step, self.n_agents), dtype=np.float32)
        self.values.fill(-100)
        self.step_count = 0
        self.obs[self.step_count] = obs
        return obs
    def get_trajectory(self):

        return (
            self.reasoning_chain, # because there is one more new line than needed
            self.obs.copy(),
            self.actions.copy(), # (max_step, n_agents (1))
            self.action_tokens.copy(),
            self.log_probs.copy(),
            self.values.copy(),
            self.problem # need to return this for generating a step accurate trajectory in AIPRM
            )
    def step(self, value, action, action_token, log_prob):
        # each of value, action and log_prob are a np (1,) array
        self.values[self.step_count] = value
        self.actions[self.step_count] = action
        self.action_tokens[self.step_count] = np.array([action_token], dtype=np.object_)
        self.log_probs[self.step_count] = log_prob
        self.step_count += 1
        # insert something here to have a version where we don't replace the self.step_tag, for AI prm labelling at the end
        action = action[0]
        action = action.replace(self.step_tag, "").strip()
        # they take out action steps here -- interesting
        print(f"action: {action}")
        self.current_state = self.current_state + action + " " + self.step_tag + "\n" # so this removes the step tags and preserves it, we want to keep using this to get the actions
        self.reasoning_chain += action + " " + self.step_tag + "\n"
        # self.current_state = self.current_state + action.strip() + "\n"
        next_obs = np.array([self.current_state], dtype=np.object_)
        self.obs[self.step_count] = next_obs
        score = 0.0
        if "step" in action.lower() or "answer" in action.lower():
            score = 1.0
        if "answer" in action.lower():
            dones = np.ones((self.n_agents), dtype=bool)
        elif self.step_count >= self.max_step: # 0 INDEXED STEPS
            dones = np.ones((self.n_agents), dtype=bool)
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
        rewards = [score for _ in range(self.n_agents)]
        infos = {"state": self.current_state}
        # nxt_obs, rewards, dones are all just some container of (1,) shape
        return next_obs, rewards, dones, infos
    