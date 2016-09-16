local learningUpdate = {
   noLearning = require 'twrl.agent.learningUpdate.noLearning',
   reinforce = require 'twrl.agent.learningUpdate.reinforce',
   tdLambda = require 'twrl.agent.learningUpdate.tdLambda',
   util = require 'twrl.agent.learningUpdate.util'
}

return learningUpdate