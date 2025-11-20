# Deep Reinforcement Learning with IsaacSim

This repository provides resources for learning and practicing Deep Reinforcement Learning using the NVIDIA IsaacSim environment.
It includes installation instructions, lecture links, and reference sources.

# Installation

## 1. Create Conda Environment

```bash
conda create -n isaacsim python=3.11
conda activate isaacsim
````

## 2. Install IsaacSim Python Package

Install IsaacSim 5.1.0 from the NVIDIA PyPI repository:

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

## 3. Install Gym

Install Gym 0.26.2 and the classic control environments:

```bash
pip install gym==0.26.2
pip install "gym[classic_control]"
```

---

## 4. Install Wandb

Install Wandb to log Training

```bash
pip install wandb
```

---

# Lecture Link

You can find the lecture videos in the playlist below:

[https://youtube.com/playlist?list=PLOYlvnpEJzrpQvlP9AqXpO5JQznzA_pjk&si=xX0VZoWrL28OiuNU](https://youtube.com/playlist?list=PLOYlvnpEJzrpQvlP9AqXpO5JQznzA_pjk&si=xX0VZoWrL28OiuNU)

---

# Source

Oh, S. (n.d.). *Deep Reinforcement Learning* [PowerPoint slides].
Department of Mathematics & Department of Data Science, Korea University.

---

# Train and Play Your Policy

## Train

```bash
python3 isaac_env/env/train.py --wandb
```

## Play

```bash
python3 isaac_env/env/paly.py --checkpoint=[your check point path]
```

```
