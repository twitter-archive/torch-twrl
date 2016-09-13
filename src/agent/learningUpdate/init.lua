local learningUpdate = {
	noLearning = require 'rl.agent.learningUpdate.noLearning',
	reinforce = require 'rl.agent.learningUpdate.reinforce',
	tdLambda = require 'rl.agent.learningUpdate.tdLambda',
	util = require 'rl.agent.learningUpdate.util'
}

return learningUpdate