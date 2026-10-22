# Dynamic Function Offloading using Deep Q-Network

This is my Master's Research Project (FA-3757) at the University of Stuttgart. The goal was to build a reinforcement learning agent that decides whether a vehicle should process a task locally or send it to a nearby Road-Side Unit (RSU).

## What this project does

Connected vehicles constantly generate tasks like object detection or path planning. These tasks need to be done fast but the vehicle itself has limited processing power. The idea here is to train a DQN agent to make this local vs offload decision smartly, based on the current state of the vehicle and the RSU.

The agent learns by trial and error over 1000 episodes, trying to minimise latency, energy use, and cost at the same time.

## How it works

The state given to the agent at each step:
- Task size (5M or 10M instructions)
- Task deadline
- Vehicle load factor (how busy the vehicle is)
- RSU load factor (how busy the RSU is)

The agent picks one of two actions: execute locally (0) or offload to RSU (1).

Reward is calculated as:
R = -(10 × latency + 5 × energy + 2 × cost)

with extra penalties for missing deadlines or overloading either side.

The network is a simple MLP: 4 → 64 → 128 → 64 → 32 → 2, trained with experience replay and a separate target network updated every 100 steps.

## Results

Training reward goes from around -6300 at the start to around -4200 by episode 1000, which shows the agent is learning. The agent consistently chooses to offload to the RSU (~1900 tasks per episode) over local execution (~700), which makes sense given the RSU has more processing power.

Test results on unseen data are consistent with training behaviour — the agent generalises well.

## Files

- `Agent.py` — DQN agent, replay memory, network definition
- `Env.py` — simulation environment (multi-RSU, vehicle movement)
- `Vehicle.py` — vehicle physics and computation model
- `RSU.py` — RSU model with Shannon capacity and Rayleigh fading
- `reward.py` — reward function
- `Dataset.py` — generates and splits the state dataset
- `Config.py` — all parameters in one place
- `main.py` — training and testing
- `pickle_plot.py` — generates plots from saved metrics
- `optimized_multi_agent_model` — saved trained model weights

## How to run
pip install torch numpy scipy tqdm matplotlib


To test with the saved model (default):
python main.py

To retrain from scratch, set `TRAIN_MODE = True` in `Config.py` and run again.
