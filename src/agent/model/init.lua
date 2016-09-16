local model = {
   mlp = require 'twrl.agent.model.mlp',
   noModel = require 'twrl.agent.model.noModel',
   qFunction = require 'twrl.agent.model.qFunction',
   tilecoding = require 'twrl.agent.model.tilecoding'
}

return model