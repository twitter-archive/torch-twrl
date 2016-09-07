local function selectAction(client, instance_id, state, envDetails, agent)
	local action
	if math.random() < agent.epsilon then
		-- Sample a random action from the environment
		action = client:env_action_space_sample(instance_id)
		-- reset eligibility traces
		local _ = agent.model.resetEligibility()
	else
		local Q = agent.model.estimateAllQ(state)
		local maxQ, maxIdx = Q:max(1)
		-- get the maximizing action
		-- actions are defined as [1,nbActions] in Lua, but [0,nbActions-1] in python
		-- need to make sure that this is accounted for
		action = maxIdx[1] - 1
		e = agent.model.e * agent.gamma * agent.lambda
	end
   -- Decay epsilon on each action selection
   if agent.epsilon > agent.epsilonMinValue then
      agent.epsilon = agent.epsilon * agent.epsilonDecayRate
   end
	return action
end
return selectAction