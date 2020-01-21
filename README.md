# Language is Power: Representing States UsingNatural Language in Reinforcement Learning

This repository accompanies the IJCAI2020 submition "Language is Power: Representing States UsingNatural Language in Reinforcement Learning". The repository contains all the code we used while writing the paper, and can be used to inspect and recreate our results. 

## Paper Abstract
Recent  advances  in  reinforcement  learning  haveshown its potential to tackle complex real life tasks. However, as  the  dimensionality  of  the  task  increases,  reinforcement  learning  methods  tend  to struggle. To  overcome  this,  we  explore  methods for representing the semantic information embedded in the state.  While previous methods focused on information in its raw form (e.g., raw visual input), we propose to represent the state using natural language. Language can represent complex scenarios and concepts,  making it a favorable candidate for representation. Empirical evidence, within the domain of ViZDoom, suggests that natural language based agents are more robust, converge faster and perform better than vision based agents, showing the benefit of using natural language representations for reinforcement learning.

## Prerequisites
- Python 3.6/3.7
- [pytorch](https://pytorch.org/)
- [gensim](https://radimrehurek.com/gensim/)
- [vizdoom](https://github.com/mwydmuch/ViZDoom)
- [baselines](https://github.com/openai/baselines)
- [pretrained glove embeddings](https://nlp.stanford.edu/projects/glove/)
  
## Features
- DQN and PPO implementations for language and vision based agents
- TextCNN, CNN PyTorch models for doom
- VizDoom gym-like environment wrapper
- Multiple natural language state generators for vizdoom
- Custom made scenarios and configuration files for vizdoom
- Word2Vec and GloVe natural language doom game state embeddigns
- plots and analysis tools for results

## Description
as described in the paper we have experimented with both PPO and DQN reinforcement learning algorithms on 3 data representations; natural languge, semantic segmentation and raw image. 
we have implemented a single script for each algorithm with a specific data representation. We used common abbreviations for the representations:
  - Natural language      : "nlp"
  - Semantic segmentation : "seg"
  - Raw image             : "viz"


For some of the basic vizdoom scenarios, we have created a noisier enviroments with varios degrees of nuisances for the learning processes. we used the following abbreviations:
 - no nuisance            : "vanilla"
 - light nuisance         : "middle"
 - heavy nuisance         : "extreme"

### Training 
For example, training a natural language based agent using PPO algorithm is acomplished by the following script:
```bash
python ppo_main_nlp.py
```
changing hyperparmeters and other options can be done by passing arguments:
```bash
python ppo_main_seg.py --scenario basic --rep-type seg --arch CONVNET_SEG --seed 6 --button_number 3 --lr 25e-5 --num-processes 32 --num-env-steps 10e6 --entropy-coef 0.01 --n-channels 1
```
In order to train multiple agents with the the same input representation on multiple seeds in parrallel:
```bash
python dqn_train_vanilla_viz.py
```
the results of the training process are PyTorch model with trained weights (.pt/.pth) files and numpy dumps of the training process.
the numpy dumps contain the achieved reward of the agent as the training progresses.
### Results
All the plotting scripts will be based on the last 5 numpy dumps, one for each seed that was trained on.

plotting basic reward as function of learning steps (agent-environment interactions):
```bash
python dqn_final_plots_script.py
```
plotting the noise robustness plots, which compare the agent's reward as a function of the nuisances in the scenario:
```bash
python dqn_noise_plots_script.py
```
comparing the achieved rewards of natural language based agent as a function of the amount of patches in the language model (described in paper):
```bash
python ppo_patches_reward_plots_script.py
```



