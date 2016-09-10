local function learn(state, action, reward, nextState, nextAction, terminal, agent)
   -- Random agent learns nothing
   local delta = 0
   if terminal then
      local _ = agent.model.resetEligibility()
   else
      local _ = agent.model.accumulateEligibility(state, action)
      -- Q-Learning
      if agent.tdLearnUpdate == 'qLearning' then
         local Q = agent.model.estimateAllQ(nextState)
         local maxQ, maxIdx = Q:max(1)
         delta = reward - agent.model.estimateQ(state, action) + (agent.gamma * maxQ[1])
      -- SARSA
      elseif agent.tdLearnUpdate == 'SARSA' then
         local Q = agent.model.estimateQ(nextState, nextAction)
         delta = reward - agent.model.estimateQ(state, action) + (agent.gamma * Q)
      end
      agent.model.w = agent.model.w + (agent.model.e * agent.alpha * delta)
   end
end
return learn
