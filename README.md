[![Build Status](https://travis-ci.com/twitter/torch-twrl.svg?token=JUyATyLn3rqyEx2nzMk9&branch=master)](https://travis-ci.com/twitter/torch-twrl) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/twitter/torch-twrl/blob/master/LICENSE)

# torch-twrl: Reinforcement Learning in Torch
torch-twrl is an RL framework built in Lua/Torch by Twitter.

Installation
------------
Install torch
~~~~
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
~~~~

Install torch-twrl
~~~~~
git clone --recursive https://github.com/twitter/torch-twrl.git
cd torch-twrl
luarocks make
~~~~~

Want to play in the gym?
------------------------
1. Start a virtual environment, not necessary but it helps keep your installation clean

2. Download and install [OpenAI Gym](https://github.com/openai/gym), gym-http-api requirements, and [ffmpeg](http://ffmpeg.org/)

~~~
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install gym
pip install -r src/gym-http-api/requirements.txt
brew install ffmpeg
~~~

Works so far?
------------------------
You should have everything you need:
* Start your gym_http_server with
~~~~
python src/gym-http-api/gym_http_server.py
~~~~

* In a new console window (or tab), run the example script (policy gradient agent in environment CartPole-v0)
~~~
cd examples
chmod u+x cartpole-pg.sh
./cartpole-pg.sh
~~~

This script sets parameters for the experiment, in detail here is what it is calling:

~~~
th run.lua \
	-env 'CartPole-v0' \
	-policy categorical \
	-learningUpdate reinforce \
   	-model mlp \
	-optimAlpha 0.9 \
   	-timestepsPerBatch 1000 \
	-stepsizeStart 0.3 \
	-gamma 1 \
	-nHiddenLayerSize 10 \
	-gradClip 5 \
	-baselineType padTimeDepAvReturn \
	-beta 0.01 \
	-weightDecay 0 \
	-windowSize 10 \
   	-nSteps 1000 \
	-nIterations 1000 \
	-video 100 \
	-optimType rmsprop \
	-verboseUpdate true \
	-uploadResults false \
	-renderAllSteps false
~~~

Your results should look something [our results from the OpenAI Gym leaderboard](https://gym.openai.com/evaluations/eval_48l1nOQ7ur6htkF9uGw)

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

3) In a new console window (or tab), run torch-twrl tests
~~~~
luarocks make; th test/test.lua
~~~~

Dependencies
------------
Testing of RL development is a tricky endeavor, it requires well established, unified, baselines and a large community of active developers. The OpenAI Gym provides a great set of example environments for this purpose. Link: https://github.com/openai/gym

The OpenAI Gym is written in python and it expects algorithms which interact with its various environments to be as well. torch-twrl is compatible with the OpenAI Gym with the use of a Gym HTTP API from OpenAI; [gym-http-api](https://github.com/openai/gym-http-api) is a submodule of torch-twrl.

All Lua dependencies should be installed on your first build.

Note: if you make changes, you will need to recompile with
~~~~
luarocks make
~~~~

## Agents
torch-twrl implements several agents, they are located in src/agents.
Agents are defined by a model, policy, and learning update.

* __Random__
	* model: noModel
	* policy: [random](https://github.com/twitter/torch-twrl/blob/master/src/agent/policy/random.lua)
	* learningUpdate: noLearning
* __TD(Lambda)__
	* model: qFunction
	* policy: [egreedy](https://github.com/twitter/torch-twrl/blob/master/src/agent/policy/egreedy.lua)
	* learningUpdate: tdLambda - implements temporal difference (Q-learning or SARSA) learning with eligibility traces (replacing or accumulating)
* __Policy Gradient__ [Williams, 1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf):
	* model: mlp - multilayer perceptron, final layeer: tanh for continuous, softmax for discrete
	* policy: [stochasticModelPolicy](https://github.com/twitter/torch-twrl/blob/master/src/agent/policy/stochasticModelPolicy.lua), normal for continuous actions, categorical for discrete
	* learningUpdate: reinforce

## Important note about agent/environment compatibility:
The OpenAI Gym has many environments and is continuously growing. Some agents may be compatible with only a subset of environments. That is, an agent built for continuous action space environments may not work if the environment expects discrete action spaces.

[Here is a useful table of the environments](https://github.com/openai/gym/wiki/Table-of-environments), with details on the different variables that may help to configure agents appropriately.

Future Work
-----------
* Automatic policy differentiation with [Autograd](https://github.com/twitter/torch-autograd)
* Parallel batch sampling
* [Additional baselines for advantage function computation](https://arxiv.org/pdf/1301.2315.pdf)
* [Cross Entropy Method (CEM)](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf)
* [Deep Q Learning (DQN)](http://arxiv.org/abs/1312.5602)
* [Double DQN](http://arxiv.org/abs/1509.06461)
* [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/pdf/1602.01783v2.pdf)
* [Deep Deterministic Policy Gradient (DDPG)](http://arxiv.org/abs/1509.02971)
* [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477v4.pdf)
* [Expected-SARSA](http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf)
* [True Online-TD](https://webdocs.cs.ualberta.ca/~sutton/papers/vSS-trueonline-ICML-2014.pdf)

References
--------------
1. Boyan, J., & Moore, A. W. (1995). Generalization in reinforcement learning: Safely approximating the value function. Advances in neural information processing systems, 369-376.
2. Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning, 3(1), 9-44.
3. Singh, S. P., & Sutton, R. S. (1996). Reinforcement learning with replacing eligibility traces. Machine learning, 22(1-3), 123-158.
4. Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. Systems, Man and Cybernetics, IEEE Transactions on, (5), 834-846.
5. Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. Vol. 1. No. 1. Cambridge: MIT press, 1998.
6. Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.

License
-------
[torch-twrl is released under the MIT License. Copyright (c) 2016 Twitter, Inc.](https://github.com/twitter/torch-twrl/blob/master/LICENSE)
