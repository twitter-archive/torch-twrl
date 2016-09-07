local function selectAction(client, instance_id, state, envDetails, agent)
   -- autocast state to a table, to handle cast to tensor
   local state = (type(state)=='number') and {state} or state
   local obsv = torch.DoubleTensor(state):reshape(1,envDetails.nbStates)
   local out = agent.model.actor:forward(obsv)
   -- Continuous action space
   -- output of the actor network should be scaled to -action_bound to action_bound
   local action = {}
   for i = 1,envDetails.nbActionSpace do
      action[i] = torch.clamp(out, envDetails.actionSpec.low[i], envDetails.actionSpec.high[i])[1][1]
   end
   return action
end
return selectAction