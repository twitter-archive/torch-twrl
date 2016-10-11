local function getPolicy(opt)
   local opt = opt or {}
   local client = opt.client
   local instanceID = instanceID
   local nStates = opt.nStates
   local model = opt.model
   local epsilon = opt.epsilon
   local gamma = opt.gamma
   local lambda = opt.lambda
   local epsilonMinValue = opt.epsilonMinValue
   local epsilonDecayRate = opt.epsilonDecayRate
   local randomActionSampler = opt.randomActionSampler

   local function selectAction(state)
      local action
      if math.random() < epsilon then
         action = randomActionSampler()
         model.eligibility:zero()
      else
         local qVals = model.estimateAllQ(state, model.weights)
         local maxQ, maxIdx = qVals:max(1)
         action = maxIdx[1] - 1
         -- actions are 0 based for gym
         model.eligibility = model.eligibility * gamma * lambda
      end
      if epsilon > epsilonMinValue then
         epsilon = epsilon * epsilonDecayRate
      end
      return action
   end
   return selectAction
end
return getPolicy