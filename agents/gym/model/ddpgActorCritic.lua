local function model(numInputs, numOutputs, agent)
	local m = {}
	nbActions = envDetails.nbActions
	-- TODO: # Final layer weights are init to Uniform[-3e-3, 3e-3]
	-- TODO: implement gradient clipping?
	
	m.actor = nn.Sequential()
		m.actor:add(nn.Linear(numInputs, 400))
		m.actor:add(nn.BatchNormalization(400))
		m.actor:add(nn.ReLU(true))
		m.actor:add(nn.Linear(400, 300))
		m.actor:add(nn.BatchNormalization(300))
		m.actor:add(nn.ReLU(true))
		m.actor:add(nn.Linear(300, numOutputs))
		m.actor:add(nn.Tanh())

	-- Critic should take in the inputs and the action and output the value of the combination
	-- action may need to skip the first layer as a design decision

	critic_encode = nn.Sequential()
		critic_encode:add(nn.Linear(numInputs, 400))
		critic_encode:add(nn.BatchNormalization(400))

	critic_action = nn.Sequential()
		critic_action:add(nn.Linear(nbActions, 400))
	
	critic_split = nn.ParallelTable()
		critic_split:add(critic_encode)
		critic_split:add(critic_action)
	
	m.critic = nn.Sequential()
		m.critic:add(critic_split)
		m.critic:add(nn.CAddTable())
		m.critic:add(nn.ReLU(true))
		m.critic:add(nn.Linear(400, 300))
		m.critic:add(nn.ReLU(true))
		m.critic:add(nn.Linear(300, 1))
		m.critic:add(nn.Tanh())

	return m
end
return model