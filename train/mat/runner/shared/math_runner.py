import time
import os
import numpy as np
from functools import reduce
import torch
from tensorboardX import SummaryWriter
from mat.agents.qwen_lora_agent import QwenLoRAgent
from mat.models.ms_prm import MSProcessRM
from mat.models.qwen_prm import QwenProcessRM
from mat.models.ai_prm import AIPRM
from mat.utils.language_buffer import LanguageBuffer, TrajectoryLanguageBuffer
from mat.trainers.llm_trainer_appo import APPOTrainer
from mat.trainers.llm_trainer_tppo import TPPOTrainer
from mat.trainers.llm_trainer_grpo import GRPOTrainer


def _t2n(x):
    return x.detach().cpu().numpy()

class MathRunner:
    def __init__(self, config):
        self.num_agents = config['num_agents']
        self.all_args = config['all_args']
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.save_interval = self.all_args.save_interval
        self.algo = self.all_args.algorithm_name
        self.prm_type = self.all_args.prm_type

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.agent = QwenLoRAgent(self.all_args.model_name_or_path, self.all_args.max_new_tokens, self.algo)
        if self.prm_type == "AI":
            self.buffer = TrajectoryLanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
        else:
            self.buffer = LanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
        
        if self.prm_type == "MS":
            self.prm = MSProcessRM(self.all_args)
        elif self.prm_type == "Qwen":
            self.prm = QwenProcessRM(self.all_args)
        elif self.prm_type == "AI":
            self.prm = AIPRM(self.all_args)
        else:
            raise NotImplementedError

        if self.algo == "APPO":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "TPPO":
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "GRPO":
            self.trainer = GRPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError

    def run(self):
        obs = self.envs.reset()
        self.buffer.obs[0] = obs.copy()
        last_obs = obs
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodic_returns = []
        for episode in range(episodes):
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads  
            # One episode is a reasoning trajectory - for PRM from AI feedback, we label the entire trajectory at the end
            if self.prm_type != "AI":
                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_tokens, log_probs = self.collect(np.concatenate(self.buffer.obs[step]))
                    
                    # output rewards
                    rewards = self.prm.get_reward(obs, actions)

                    # Obs reward and next obs
                    obs, fake_rewards, dones, infos = self.envs.step(actions)

                    # insert data into buffer
                    data = obs, rewards, dones, values, actions, action_tokens, log_probs
                    self.insert(data)
                    
                    for i in range(self.n_rollout_threads):
                        if dones[i, 0]:
                            episodic_returns.append(rewards[i, 0])
                            # add trajectories here
                self.before_update()
                train_infos = self.trainer.train(self.buffer)      
                self.buffer.after_update()
            else:
                for step in range(self.episode_length):
                    # Sample actions
                    
                    values, actions, action_tokens, log_probs = self.collect(np.concatenate(last_obs)) # change this for trajectory based
                    
                    # output rewards
                    # rewards = self.prm.get_reward(obs, actions)

                    # Obs reward and next obs
                    last_obs, rewards, dones, infos = self.envs.extended_step(values, actions, action_tokens, log_probs) # annoying there here -- if it's done it's going to automaitcally reset
                    # don't insert data, we're going to collect all this data for the n_rollout_threads. Then everytime one of them is done, we need to take the trajectory... maybe this is better to implement in subprocenv
                    # data = obs, rewards, dones, values, actions, action_tokens, log_probs
                    # self.insert(data)
                    done_reasonings, actions, action_tokens, log_probs, values, traj_obs = [], [], [], [], [], []
                    problems = []
                    for i in range(self.n_rollout_threads):
                        if dones[i, 0]: # dones, and any other thing returned from self.envs.step is going to be in the shape num_rollouts, num_agents
                            # episodic_returns.append(rewards[i, 0])
                            reasoning_chain, obs, action, action_token, log_prob, value, problem = infos[i]['trajectory'] # remember these are ALREADY COPIED NP ARRAYS of shape (max_steps (episode_length), n_agents (1))
                            # action_token = self.pad_left(action_tokens, self.episode_length, pad_value=-100)
                            print("Shapes", obs.shape, action.shape, action_token.shape, log_prob.shape, value.shape)
                            actions.append(action)
                            action_tokens.append(action_token)
                            log_probs.append(log_prob)
                            values.append(value)
                            traj_obs.append(obs)
                            done_reasonings.append(reasoning_chain)
                            problems.append(problem)
                            # get the rewards here
                            # run it on trajectories?
                            # add trajectories here
                    # run a loop here with self.insert(). some notes > it's on every episode, we can delay it all to the end and add everything then? Or how do we still create accurate trajectory buffers because we can't ensure
                    # we can make sure to only take the last (max_trajectories) in the buffer when doing the buffer adding, by making a rotating queue
                    # empty out buffer after the update and start again.
                    # max trajectories should be the number of rollout_threads * 1.5, to ensure some lenience inside each episode
                    if len(done_reasonings) > 0:

                        rewards = self.prm.get_reasoning_traj_reward(done_reasonings, problems) # (num_trajectories, max_length) numpy array
                        for ind in range(len(done_reasonings)):
                            if rewards[ind].shape[0] < self.episode_length:
                                reward = np.pad(rewards[ind], self.episode_length, constant_values=-100)
                            else:
                                reward = rewards[ind][:self.episode_length]
                            reward = np.expand_dims(reward, axis=-1)
                            self.insert_trajectory((traj_obs[ind], reward, values[ind], actions[ind], action_tokens[ind], log_probs[ind])) # maintain buffer size by FIFO RULE
                        
                self.before_update()
                train_infos = self.trainer.train(self.buffer)
                self.buffer.after_update()

                # train here + empty buffer

                
                        
                # nice thing about having full trajectories: V_{t+1} is guaranteed to be 0
            # compute return and update network


            
            
            # save model
            if (episode == episodes - 1 or episode % self.save_interval == 0):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                print("average_step_rewards: ", np.mean(self.buffer.rewards))
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                train_infos["average_currect_rate"] = np.mean(episodic_returns)
                self.log_infos(train_infos, total_num_steps)
                episodic_returns = []

            # eval
            # if self.all_args.use_eval and episode % self.eval_interval == 0:
            #     self.eval(total_num_steps)
        

    @torch.no_grad()
    def collect(self, recent_obs): # collects agent actions for N envs
        print(recent_obs.shape)
        behaviour_data = self.agent.infer_for_rollout(recent_obs)
        
        actions, action_tokens, values, log_probs = behaviour_data
        
        # [self.envs, agents]
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs
    
    @torch.no_grad()

    def insert(self, data):
        obs, rewards, dones, values, actions, action_tokens, log_probs = data
        # num agents is always one anyways?
        dones_env = np.all(dones, axis=1) # gives indices of envs that are done
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32) # create mask of size (# of envs, # of agents (1))
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32) # 
        if self.algo == "APPO" or self.algo == "GRPO":
            self.buffer.insert_appo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        elif self.algo == "TPPO":
            self.buffer.insert_tppo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        else:
            raise NotImplementedError
    def insert_trajectory(self, data):
        obs, rewards, values, actions, action_tokens, log_probs = data
        print("traj shapes", obs.shape, rewards.shape, values.shape, actions.shape, action_tokens.shape, log_probs.shape)
        # num agents is always one anyways?
        if self.algo == "APPO" or self.algo == "GRPO":
            self.buffer.insert_appo(obs, actions, values, rewards, action_tokens, log_probs)
        elif self.algo == "TPPO":
            self.buffer.insert_tppo(obs, actions, values, rewards, action_tokens, log_probs)
        else:
            raise NotImplementedError

    def pad_left(self, array, max_length, pad_value=0):
        """Pad an array on the left to reach the specified maximum length."""
        pad_width = max_length - array.shape[0]
        if pad_width <= 0:
            return array  # No padding needed
        return np.pad(array, (pad_width, 0), mode='constant', constant_values=pad_value)

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data.""" # (through bellman backup)
        next_values = -100
        if self.prm_type != "AI":
            next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[-1]))
            next_values = np.array(np.split(next_values, self.n_rollout_threads))
        if self.algo == "APPO":
            self.buffer.batch_process_appo(next_values)
        elif self.algo == "TPPO":
            self.buffer.batch_process_tppo(next_values)
        elif self.algo == "GRPO":
            self.buffer.batch_process_grpo()
        else:
            raise NotImplementedError

    def log_infos(self, infos, total_num_steps):
        for k, v in infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episodic_returns = []

        eval_obs = self.eval_envs.reset()
        while True:
            eval_actions, _ = self.agent.get_actions(np.concatenate(eval_obs))
            eval_actions = np.array(np.split(eval_actions, self.n_eval_rollout_threads))
            eval_obs, eval_rewards, eval_dones, _ = self.eval_envs.step(eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones[eval_i, 0]:
                    eval_episode += 1
                    eval_episodic_returns.append(eval_rewards[eval_i])

            if eval_episode >= self.all_args.eval_episodes:
                eval_currect_rate = np.mean(eval_episodic_returns)
                env_infos = {'eval_currect_rate': eval_currect_rate}     
                print("total_num_steps: ", total_num_steps)
                print("eval_currect_rate is {}.".format(eval_currect_rate))           
                self.log_infos(env_infos, total_num_steps)
                break
                
    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.agent.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.agent.restore(model_dir)



