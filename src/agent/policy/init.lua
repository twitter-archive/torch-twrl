local policy = {
	egreedy = require 'rl.agent.policy.egreedy',
	normal = require 'rl.agent.policy.normal',
	categorical = require 'rl.agent.policy.categorical',
	random = require 'rl.agent.policy.random',
}

return policy