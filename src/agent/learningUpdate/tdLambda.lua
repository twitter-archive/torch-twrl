local function getLearningUpdate(opt)
   local opt = opt or {}
   local model = opt.model
   local envDetails = opt.envDetails
   local tdLearnUpdate = opt.tdLearnUpdate
   local gamma = opt.gamma
   local alpha = opt.stepsizeStart
   local relativeAlpha = opt.relativeAlpha
   local numTilings = opt.numTilings
   if relativeAlpha ~= 0 then
      alpha = relativeAlpha / numTilings
   end

   local function learn(transition, nIter)
      local state = transition.state
      local action = transition.action
      local reward = transition.reward
      local nextState = transition.nextState
      local nextAction = transition.nextAction
      local terminal = transition.terminal
      local delta = 0
      if terminal == true then
         delta = reward - model.estimateQ(state, action, model.weights)
         model.weights = model.weights + (alpha * delta * model.eligibility)
         model.eligibility:zero()
      else
         model.accumulateEligibility(state, action, model.eligibility)
         if tdLearnUpdate == 'qLearning' then
            local qVals = model.estimateAllQ(nextState, model.weights)
            local maxQ, maxIdx = qVals:max(1)
            delta = reward + (gamma * maxQ[1]) - model.estimateQ(state, action, model.weights)
         elseif tdLearnUpdate == 'SARSA' then
            local qVal = model.estimateQ(nextState, nextAction, model.weights)
            delta = reward + (gamma * qVal) - model.estimateQ(state, action, model.weights)
         end
         model.weights = model.weights + (alpha * delta * model.eligibility)
      end
   end
   return learn
end
return getLearningUpdate