local agent = {
   baseAgent = require 'twrl.agent.baseAgent',
   learningUpdate = require 'twrl.agent.learningUpdate',
   model = require 'twrl.agent.model',
   policy = require 'twrl.agent.policy'
}

return agent