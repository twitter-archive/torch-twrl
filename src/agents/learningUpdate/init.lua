local learningUpdate = {
	noLearning = require 'rl.agents.learningUpdate.noLearning',
	reinforce = require 'rl.agents.learningUpdate.reinforce',
	tdLambda = require 'rl.agents.learningUpdate.tdLambda',
}

return learningUpdate