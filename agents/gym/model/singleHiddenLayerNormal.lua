local function model(numInputs, numOutputs, agent)
   return nn.Sequential():add(nn.Linear(numInputs, agent.nHiddenLayerSize)):add(nn.Tanh()):add(nn.Linear(agent.nHiddenLayerSize, numOutputs)):add(nn.Tanh())
end
return model