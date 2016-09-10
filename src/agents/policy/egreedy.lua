local function selectAction(client, instanceID, state, envDetails, agent)
	local action
	if math.random() < agent.epsilon then
		-- Sample a random action from the environment
		action = client:env_action_space_sample(instanceID)
		-- reset eligibility traces
		local _ = agent.model.resetEligibility()
	else
		-- get the maximizing action
		local Q = agent.model.estimateAllQ(state)
		local maxQ, maxIdx = Q:max(1)
		-- actions are defined as [1,nbActions] in agent, but [0,nbActions-1] in environments
		action = maxIdx[1] - 1
		-- decay the eligibility traces
		agent.model.e = agent.model.e * agent.gamma * agent.lambda
	end
   -- Decay epsilon on each action selection
   if agent.epsilon > agent.epsilonMinValue then
      agent.epsilon = agent.epsilon * agent.epsilonDecayRate
   end
	return action
end
return selectAction