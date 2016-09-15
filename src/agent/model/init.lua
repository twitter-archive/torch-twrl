local model = {
   mlp = require 'rl.agent.model.mlp',
   noModel = require 'rl.agent.model.noModel',
   qFunction = require 'rl.agent.model.qFunction',
   tilecoding = require 'rl.agent.model.tilecoding'
}

return model