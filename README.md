# Proto-RL: Reinforcement Learning with Prototypical Representations

This is a PyTorch implementation of **Proto-RL** from

**Reinforcement Learning with Prototypical Representations** by

[Denis Yarats](https://cs.nyu.edu/~dy1042/), [Rob Fergus](https://cs.nyu.edu/~fergus/pmwiki/pmwiki.php), [Alessandro Lazaric](http://chercheurs.lille.inria.fr/~lazaric/Webpage/Home/Home.html), [Lerrel Pinto](https://cs.nyu.edu/~lp91/).

[[Paper]](https://arxiv.org/abs/2102.11271)

## Citation
If you use this repo in your research, please consider citing the paper as follows
```
@article{yarats2021proto,
    title={Reinforcement Learning with Prototypical Representations},
    author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
    year={2021},
    eprint={2102.11271},
    archivePrefix={arXiv},
    primaryClass={cs.ML}
}
```

## Requirements
We assume you have access to a gpu that can run CUDA 11. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```
conda activate proto
```

## Instructions
In order to pretrain the agent you need to specify the number of task-agnostic environment steps by setting `num_expl_steps`, after that many steps, the agent will start receving the downstream task reward until it takes `num_train_steps` in total. For example, to pre-train the Proto-RL agent on `Cheetah Run` task unsupervisely for 500k environment steps and then train it further with the downstream reward for another 500k steps, you can run:
```
python train.py env=cheetah_run num_expl_steps=250000 num_train_steps=500000
```
Note that we divide the number of steps by action repeat, which is set to 2 for all the environments.

This will produce the `exp_local` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. To launch tensorboard run
```
tensorboard --logdir exp_local
```
