local function selectAction(client, instance_id, state, envDetails, agent)
   -- Sample a random action from the environment
   return client:env_action_space_sample(instance_id)
end
return selectAction