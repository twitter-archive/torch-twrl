local function selectAction(client, instance_id, state, envDetails, agent)
   -- autocast state to a table, to handle cast to tensor
   local state = (type(state)=='number') and {state} or state
   local obsv = torch.DoubleTensor(state):reshape(1,envDetails.nbStates)
   local out = agent.model:forward(obsv)
   -- Continuous action space
   --- Action selection policy is based on a sample from the normal distribution centered around the output from the network
   local action = torch.DoubleTensor(envDetails.nbActionSpace):zero()
   for i = 1,envDetails.nbActionSpace do
      -- action is a random real number, according to the normal distribution, with the given mean and stdv, and then scaled into the action space range
      action[i] = torch.normal(out,agent.policyStd) * (envDetails.actionSpaceBounds[i][3]/2)
   end
   atable = {}
   for i =1,action:size(1) do
      atable[i] = action[i]
   end
   return atable
end
return selectAction