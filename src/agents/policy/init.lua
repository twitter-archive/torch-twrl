local policy = {
	egreedy = require 'rl.agents.policy.egreedy',
	normal = require 'rl.agents.policy.normal',
	categorical = require 'rl.agents.policy.categorical',
	random = require 'rl.agents.policy.random',
}

return policy