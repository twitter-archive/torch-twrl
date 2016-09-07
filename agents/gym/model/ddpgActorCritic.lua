local function model(numInputs, numOutputs, agent)
	local m = {}
	nbActions = envDetails.nbActions
	-- TODO: # Final layer weights are init to Uniform[-3e-3, 3e-3]
	m.actor = nn.Sequential():add(nn.Linear(numInputs, 400)):add(nn.ReLU()):add(nn.Linear(400, 300)):add(nn.ReLU()):add(nn.Linear(300, numOutputs)):add(nn.Tanh())
	-- Critic should take in the inputs and the action and output the value of the combination
	-- action may need to skip the first layer as a design decision
	print(numInputs,nbActions)
	m.critic = nn.Sequential():add(nn.Linear(numInputs+nbActions, 400)):add(nn.ReLU()):add(nn.Linear(400, 300)):add(nn.ReLU()):add(nn.Linear(300, 1)):add(nn.Tanh())
	return m
end
return model