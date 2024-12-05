from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from torch import nn
import numpy as np
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE

# EleutherAI/llemma_34b
class LlemmaPRM(nn.Module):

    def __init__(self, all_args):
        super().__init__()
        self.model_name_or_path = all_args.prm_model_name_or_path # ../../models/llemma_34b
        self.prm_checkpoint_path = all_args.prm_checkpoint_path
        print(f"prm_base_model_path: {self.model_name_or_path}")
        print(f"prm_checkpoint_path: {self.prm_checkpoint_path}")
        
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_eos_token=False, padding_side='left')
        self.tokenizer.pad_token_id = 0 # "<|image_pad|>"
        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}") # [id for good, id for bad token]
        print(self.candidate_tokens)
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1] # [find the id for the step token (self.step_tag)]
        print(self.step_tag_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, 
                                                          device_map="auto", 
                                                          torch_dtype=torch.bfloat16,
                                                        #   attn_implementation="flash_attention_2",
                                                          ).eval()
        # adapter_config = PeftConfig.from_pretrained(cp_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model, self.prm_checkpoint_path)
        
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
