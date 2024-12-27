from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from torch import nn
import numpy as np
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE
import re

IN_CONTEXT_EXAMPLE = """
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Step 1: Janet's ducks lay 16 eggs per day. ки
Step 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки
Step 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки
Step 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. ки
Step 5: The answer is: 18. ки

### Step 1: Score 3
The statement correctly identifies the total number of eggs laid by Janet's ducks per day.
### Step 2: Score 5
The calculation is correct: subtracting the 3 eggs eaten for breakfast from the total 16 eggs results in 13 eggs remaining.
### Step 3: Score 5
The calculation is correct: subtracting the 4 eggs used for baking muffins from the remaining 13 eggs results in 9 eggs remaining.
### Step 4: Score 5
The calculation is correct: multiplying the 9 remaining eggs by the price of $2 per egg results in $18.
### Step 5: Score 5
The final answer is correct: Janet makes $18 every day at the farmers' market.
However, since the problem states that the final answer is incorrect, we need to re-evaluate the steps to identify any potential logical or arithmetic errors. Upon re-evaluation, all steps appear to be logically sound and mathematically correct. Therefore, the issue might lie outside the provided steps, such as a misinterpretation of the problem statement or an external factor not considered in the reasoning.
"""

LABELLING_PROMPT = """Now you are an expert mathmatician, you are going to evaluate the reasoning steps of a problem. Here is a reasoning of a math problem. 
Score each reasoning step from -5 to 5. Please carefully check the calculation and logic. 
A step can achieve high score if it contribute to the final solution (i.e. removing it will likely result in incorrect answer). 
If a step is correct but you think is redundant and not necessary for the final solution, you can give a small negative score. Every reasoning step is a sentence.  
This problem is known to have incorrect answer, which means at least one reasoning step is incorrect. If an answer is not provided, use your best judgement.
For each step, output your answer in the format: ### Step [Step Number]: Score [Choose any number between -5 to 5 to rate the step]. An example step score is: ### Step 3: Score 2.
"""


class AIPRM(nn.Module):

    def __init__(self, all_args):
        super().__init__()
        self.model_name_or_path = all_args.prm_model_name_or_path # ../../models/llemma_34b
        self.prm_checkpoint_path = all_args.prm_checkpoint_path
        print(f"prm_base_model_path: {self.model_name_or_path}")
        print(f"prm_checkpoint_path: {self.prm_checkpoint_path}")
        
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'
        self.max_tokens = all_args.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, padding_side='left')
        self.tokenizer.pad_token_id = 0 # "<|image_pad|>"
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1] # [find the id for the step token (self.step_tag)]
        print(self.step_tag_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map='auto', torch_dtype=torch.bfloat16).eval()
        # self.model = AutoModelForCausalLM.from_pretrained(
        #                                             self.model_name_or_path,
        #                                             torch_dtype="auto",
        #                                             device_map="auto").eval()
        # adapter_config = PeftConfig.from_pretrained(cp_path)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model, self.prm_checkpoint_path)
        
    @torch.no_grad()
    def get_reward(self, obs: list[np.ndarray[str]], actions: list[np.ndarray[str]]): # the reason it's a list is because we have N parallel vectorized envs
        inputs_for_prm = []
        for o, a in zip(obs.copy(), actions.copy()): # this loop is just basically input processing
            o = o[0].replace(IN_CONTEXT_EXAMPLE, "") # removes the in-context example in the prompt
            a = a[0].replace(self.step_tag, "").strip() # removes the ки in the output (that the policy is conditioned to output)?
            inputs_for_prm.append(f"{o}{a} {self.step_tag}") # inserts it back into this format for the PRM, in this case it's LLEMMA, which isn't finetuned so we need a different step_tag
        # inputs_for_prm is a list of length # of envs, with each one having the current process
        input_ids = self.tokenizer(inputs_for_prm, return_tensors="pt", padding=True).to("cuda")
        # self.model(**input_ids) -> outputs the a batch style, where it's (# steps, length answer)
        logits = self.model(**input_ids).logits[:, :, self.candidate_tokens] # extrack logits for the good and bad tokens, what is the shape of this?
        score = logits.softmax(dim=-1)[:, :, 0]
        
        step_scores = []
        for i in range(np.shape(score)[0]): # length of processes
            step_score = score[i][input_ids["input_ids"][i] == self.step_tag_id]
            last_step_score = step_score[-1]
            step_scores.append([last_step_score.item()])
        step_scores = np.array(step_scores)
        
        return step_scores
    
    @torch.no_grad()
    def get_reasoning_traj_reward(self, trajectories : list[str], problems : list[str]):
        # This should take in an entire step trajectory for each of the 
        # trajectory is a list of strings of the entire model prompt separated by \n
        templated_trajs = []
        max_traj_length = 0
        for traj, problem in zip(trajectories, problems): 
            steps = traj[len(problem) + 1:].replace('\n', ' ').split('ки')[:-1] # split up steps, take out the last dummy empty string
            new_traj = '\n'.join([f"Step {ind + 1}: {steps[ind]} ки"for ind in range(len(steps))])
            new_traj = problem + '\n' + new_traj
            # print(new_traj)
            max_traj_length = max(len(new_traj.split(' ')), max_traj_length)
            conversation = [
                {
                    "role" : "system",
                    "content" : LABELLING_PROMPT
                },
                {
                    "role" : "user",
                    "content" :  new_traj
                }
            ]
            templated_trajs.append(self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
        
        tokenized_chats = self.tokenizer(
            templated_trajs,
            padding=True,
            return_tensors="pt"
        ).to('cuda')
        
        output_ids = self.model.generate(
            **tokenized_chats,
            do_sample=False,
            max_new_tokens=max_traj_length
        )
        outputs_trimmed = [outputs[inputs.shape[0]:] for inputs, outputs in zip(tokenized_chats.input_ids, output_ids)]
        decoded_outputs = self.tokenizer.batch_decode(
            outputs_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ) # returns a list of each of the answers
        # print(decoded_outputs)
        # run regex extraction
        pattern = r"### Step (\d+): Score (-?\d+)"
        rewards = []
        for ind, output in enumerate(decoded_outputs):
            # how many new lines are there in this?
            num_steps = trajectories[ind].count(self.step_tag) # count how many steps there were in the actual trajectory
            matches = re.findall(pattern, output)
            print(f"num steps {num_steps}, decoded output {output}")
            print(matches)
            if len(matches) != num_steps:
                print("ERROR: AI SCORER FAILED TO LABEL ALL STEPS")
            scores = np.array([float(match[1]) for match in matches])
            mean, std = np.mean(scores), np.std(scores)
            scores = (scores - mean) / (std + 1e-7)
            rewards.append(scores)
        
        return rewards
            # what do we do in the case that a step doesn't have a reward? 
            # regex extraction on this

    # takes a list of trajectories and corresponding problems and calculates reward
    # using AI scoring. Prompts the AI n times and returns reward as the average over all
    # queries. To add a variance penalty, specify var_penalty
    def get_reasoning_traj_reward_avg(self, trajectories: list[str], problems: list[str], n: int,
                                      var_penalty: float = 0.0):
        runs = []
        for i in range(n):
            runs.append(self.get_reasoning_traj_reward(trajectories, problems))
        result = []
        for j in range(len(runs[0])):
            curr_trajectories = [run[j] for run in runs]
            result.append(np.mean(curr_trajectories, axis=0) - var_penalty * np.var(curr_trajectories, axis=0))
        return result