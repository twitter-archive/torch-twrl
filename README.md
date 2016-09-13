# torch-rl: Reinforcement Learning in Torch
torch-rl is an RL framework built in Lua/Torch by Twitter.

Installation
------------
Clone from the repository, and install torch-rl:
~~~~~
git clone --recursive https://github.com/twitter/torch-rl.git
cd torch-rl
luarocks make
~~~~~

Want to play in the gym?
------------------------
1) Start a virtual environment (not necessary but helps keep everything clean)
2) Download and install [OpenAI Gym](https://github.com/openai/gym)
~~~
virtualenv venv
pip install gym
~~~

Works so far? 
------------------------
You should have everything you need:
* Start your gym_http_server with 
~~~~
python src/gym-http-server/gym_http_server.py
~~~~

* In a new console window (or tab), run the example script (policy gradient agent in environment CartPole-v0)
~~~
chmod u+x run_tests.sh
./run_tests.sh
~~~

This script sets parameters for the experiment, in detail here is what it is calling:

~~~
th testScript.lua -env 'CartPole-v0' \
	-policy categorical -learningUpdate reinforce \
	-model mlp -optimAlpha 0.9 \
 -timestepsPerBatch 1000 -stepsizeStart 0.3 -gamma 1 \
	-nHiddenLayerSize 10 -gradClip 5 -baselineType padTimeDepAvReturn \
	-beta 0.01 -weightDecay 0 -windowSize 100 \
 -nSteps 1000 -nIterations 1000 -video 0 \
	-uploadResults true -renderAllSteps false
~~~
	
Your results should look something [our results from the OpenAI Gym leaderboard](https://gym.openai.com/evaluations/eval_9wwlxNWeTOWaFJcwioF0zQ):

Doesn't work?
------------------------
1) Test the gym-http-api
~~~~
cd /src/gym-http-api/
nose2
~~~~

2) Start a Gym HTTP server in your virtual environment
~~~~
python src/gym-http-api/gym_http_server.py
~~~~

3) In a new console window (or tab), run torch-rl tests
~~~~
luarocks make; th test/test.lua
~~~~

Dependencies
------------
Testing of RL development is a tricky endeavor, it requires well established, unified, baselines and a large community of active developers. The OpenAI Gym provides a great set of example environments for this purpose. Link: https://github.com/openai/gym

The OpenAI Gym is written in python and it expects algorithms which interact with its various environments to be as well. torch-rl is compatible with the OpenAI Gym with the use of a modified Gym HTTP API, based on the original code from OpenAI; [gym-http-api](https://github.com/korymath/gym-http-api) is a submodule of torch-rl.

All Lua dependencies should be installed on your first build.

Note: if you make changes, you will need to recompile with
~~~~
luarocks make
~~~~


Code Structure
--------------
The torch-rl directory structure is laid out as follows:

* agents - contains all the RL agents in torch-rl
* local - agents for testing local environments
* gym - agents for testing gym environments
* envs - local environments
* testing - framework and scripts for testing 
* gym - gym specific testing
* experiment_gym.lua - framework for gym testing
run_tests.sh - bash script for running multiple gym tests
test_gym.lua - main testing function for gym tests
local - local specific testing
experiment_local.lua - framework for local testing 
run_tests.sh - bash script for running multiple local tests
test_local.lua - main testing function for local tests
util
gym_utilities.lua - gym agent and environment supplementary functions, also contains baseline calculations
tilecoding.lua - function approximation, based on TileCoding-v3 from Rich Sutton
trajectory.lua - for storing trajectories in batch training methods
index.rst - general README for the project
 
## Agents

torch-rl implements several agents, they are located in /agents:

### Gym agents
* Random
* TD(Lambda) - implements temporal difference learning with eligibility traces
* Q-learning and SARSA-learning
* Eligibility traces can be replacing or accumulating
* REINFORCE [Williams, 1992] implements vanilla policy gradient
** For continuous action spaces, a normal policy is used
** For discrete action spaces, a categorical policy is used

### Local agents, for the local environments:
* randomDiscrete.lua - Random agent which handles discrete environments
* sarsa.lua - tabular on-policy SARSA, for 1D or 2D discrete state spaces
* sarsaLambda.lua - linear function approximation on-policy SARSA, with eligibility traces
* qLambda.lua - linear function approximation off-policy Q-learning, with eligibility traces

## Note on OpenAI Gym environments:

The OpenAI Gym has many environments and is continuously growing
Some agents may be compatible with only a subset (agent built for continuous action space environments may not work if the environment expects discrete action spaces)
Here is a useful table of the environments, with details on the different variables that may help to configure agents appropriately.

## Environments

### Local environments
torch-rl includes several local environments for testing, they are located in /envs.

### OpenAI Gym environments
The OpenAI gym has many environments, agents may be compatible with only a subset.
Here is a table of the environments, with details on the different variables therein https://github.com/openai/gym/wiki/Table-of-environments. It will help you configure your agents appropriately.

Future Work
-----------
* Autograd Integration for automatic policy differentiation
* Actor-Critic Policy Gradient
* Cross Entropy Method
* Parallel workers for random environment sampling
* A3C
* DDPG - Link1
* TRPO
* PAL
* Bayesian RL - more background
* Expected SARSA (van Seijen 2009 [PDF])
* True Online TD (van Seijen, Sutton 2014 [PDF])

References
--------------
1. Boyan, J., & Moore, A. W. (1995). Generalization in reinforcement learning: Safely approximating the value function. Advances in neural information processing systems, 369-376.
2. Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning, 3(1), 9-44.
3. Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. Machine learning, 22(1-3), 123-158.
4. Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. Systems, Man and Cybernetics, IEEE Transactions on, (5), 834-846.
5. Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. Vol. 1. No. 1. Cambridge: MIT press, 1998.
6. Some environment code based on: https://github.com/Kaixhin/rlenvs
