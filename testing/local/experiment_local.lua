local function experiment(env, agent, nSteps, nEpisodes, nReport)
   -- Initialize the experiment variables
   local reward, terminal
   local episodes, epReward, totalReward = 0, 0, 0
   local stepReward = torch.Tensor(1):zero()
   local epRewards = torch.Tensor(nEpisodes):zero()
   for nEp = 1, nEpisodes do
      -- reset the episode reward and sum of discounted reward
      epReward = 0
      -- Initial state and action of episode
      local state = env.start()
      local action = agent.selectAction(state)
      -- Step through the interaction
      for i = 1, nSteps do
         -- Take action, observe reward and next state
         local reward, stateNext, terminal = env.step(action)
         -- Keep track of step and episode reward
         stepReward = stepReward:cat(torch.DoubleTensor(1):fill(reward))
         epReward = epReward + reward
         -- pass a terminal flag if episode complete
         if i == nSteps then
            terminal = true
         end
         -- Perform the learning update with the selected next action if needed
         local actionNext = agent.selectAction(stateNext)
         local updatedActionChoice = agent.learn(state, action, reward, stateNext, actionNext, terminal)
         -- If reached a terminal state, start the next episode
         if terminal then
            -- print('done in: ' .. i)
            episodes = episodes + 1
            state = env.start()
            break
         end
         -- Update state and action
         -- agent learning should pass through the next action
         if updatedActionChoice ~= nil then
            action = updatedActionChoice
         else
            action = actionNext
         end
         state = stateNext
      end
      -- collect episode reward
      epRewards[nEp] = epReward
      -- reset the step reward
      stepReward = torch.DoubleTensor(1):zero()
      -- accumulate total reward
      totalReward = totalReward + epReward
      -- report reward
      if nEp % nReport == 0 then
         -- print('Ep ' .. nEp .. ' reward: ' .. epReward)
         print('Ep: ' .. nEp .. ' average reward over last ' .. nReport .. ' episodes: ' .. epRewards[{{nEp-nReport+1,nEp}}]:mean())
      end
   end
   return true
end
return experiment