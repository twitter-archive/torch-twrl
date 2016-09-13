local actionSamplers = require 'rl.agent.policy.actionSamplers'

local policy = {
	 egreedy = require 'rl.agent.policy.egreedy',
	 normal = require 'rl.agent.policy.stochasticModelPolicy'({
				 actionSampler = actionSamplers.normal
		  }),
	 categorical = require 'rl.agent.policy.stochasticModelPolicy'({
				 actionSampler = actionSamplers.categorical
		  }),
	 random = require 'rl.agent.policy.random',
}

return policy
