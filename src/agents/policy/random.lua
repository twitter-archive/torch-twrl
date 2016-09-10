local function selectAction(client, instanceID, state, envDetails, agent)
   -- Sample a random action from the environment
   return client:env_action_space_sample(instanceID)
end
return selectAction