import os
os.environ['MUJOCO_GL'] = 'egl'

import copy
import math
import pickle as pkl
import sys
import time

import numpy as np

import dmc
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        self.buffer_dir = utils.make_dir(self.work_dir, 'buffer')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = dmc.make(cfg.env, cfg.frame_stack, cfg.action_repeat,
                            cfg.seed)
        self.eval_env = dmc.make(cfg.env, cfg.frame_stack, cfg.action_repeat,
                                 cfg.seed + 1)

        obs_spec = self.env.observation_spec()['pixels']
        action_spec = self.env.action_spec()

        cfg.agent.params.obs_shape = obs_spec.shape
        cfg.agent.params.action_shape = action_spec.shape
        cfg.agent.params.action_range = [
            float(action_spec.minimum.min()),
            float(action_spec.maximum.max())
        ]
        # exploration agent uses intrinsic reward
        self.expl_agent = hydra.utils.instantiate(cfg.agent,
                                                  task_agnostic=True)
        # task agent uses extr extrinsic reward
        self.task_agent = hydra.utils.instantiate(cfg.agent,
                                                  task_agnostic=False)
        self.task_agent.assign_modules_from(self.expl_agent)

        if cfg.load_pretrained:
            pretrained_path = utils.find_pretrained_agent(
                cfg.pretrained_dir, cfg.env, cfg.seed, cfg.pretrained_step)
            print(f'snapshot is taken from: {pretrained_path}')
            pretrained_agent = utils.load(pretrained_path)
            self.task_agent.assign_modules_from(pretrained_agent)

        # buffer for the task-agnostic phase
        self.expl_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device)
        # buffer for task-specific phase
        self.task_buffer = ReplayBuffer(obs_spec.shape, action_spec.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device)

        self.eval_video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def get_agent(self):
        if self.step < self.cfg.num_expl_steps:
            return self.expl_agent
        return self.task_agent

    def get_buffer(self):
        if self.step < self.cfg.num_expl_steps:
            return self.expl_buffer
        return self.task_buffer

    def evaluate(self):
        avg_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            time_step = self.eval_env.reset()
            self.eval_video_recorder.init(enabled=(episode == 0))
            episode_reward = 0
            episode_success = 0
            episode_step = 0
            while not time_step.last():
                agent = self.get_agent()
                with utils.eval_mode(agent):
                    obs = time_step.observation['pixels']
                    action = agent.act(obs, sample=False)
                time_step = self.eval_env.step(action)
                self.eval_video_recorder.record(self.eval_env)
                episode_reward += time_step.reward
                episode_step += 1

            avg_episode_reward += episode_reward
            self.eval_video_recorder.save(f'{self.step}.mp4')
        avg_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', avg_episode_reward, self.step)
        self.logger.dump(self.step, ty='eval')

    def run(self):
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        done = True
        while self.step <= self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()
                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(self.step, ty='train')

                time_step = self.env.reset()
                obs = time_step.observation['pixels']
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            agent = self.get_agent()
            replay_buffer = self.get_buffer()
            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode - 1, self.step)
                self.evaluate()

            # save agent periodically
            if self.cfg.save_model and self.step % self.cfg.save_frequency == 0:
                utils.save(
                    self.expl_agent,
                    os.path.join(self.model_dir, f'expl_agent_{self.step}.pt'))
                utils.save(
                    self.task_agent,
                    os.path.join(self.model_dir, f'task_agent_{self.step}.pt'))
            if self.cfg.save_buffer and self.step % self.cfg.save_frequency == 0:
                replay_buffer.save(self.buffer_dir, self.cfg.save_pixels)

            # sample action for data collection
            if self.step < self.cfg.num_random_steps:
                spec = self.env.action_spec()
                action = np.random.uniform(spec.minimum, spec.maximum,
                                           spec.shape)
            else:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=True)

            agent.update(replay_buffer, self.step)

            time_step = self.env.step(action)
            next_obs = time_step.observation['pixels']

            # allow infinite bootstrap
            done = time_step.last()
            episode_reward += time_step.reward

            replay_buffer.add(obs, action, time_step.reward, next_obs, done)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
