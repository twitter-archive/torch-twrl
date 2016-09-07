-- SARSALambda with linear function approximation with tile coding
local function getAgent(opt)
   local opt = opt or {}
   local env = opt.env or nil
   -- Tile-coding parameters
   local numTiles = opt.numTiles -- in each direction
   local memorySize = numTiles * numTiles
   local numTilings = opt.numTilings -- stacked offset tiligns
   -- State space dependent parameters
   local stateSpec = env.getStateSpec()
   local stateScalingFactor = {}
   local stateMins = {}
   local stateMaxs = {}
   if stateSpec[1][2] > 1 then
      -- handle the large embedding state space for Favorites
      for j = 1, stateSpec[1][2] do
         stateScalingFactor[j] = 1 / numTilings
      end
   else
      for i = 1,#stateSpec do
         stateMins[i] = stateSpec[i][3][1]
         stateMaxs[i] = stateSpec[i][3][2]
         if stateMins[i] and stateMaxs[i] then
            stateScalingFactor[i] = (stateMaxs[i] - stateMins[i]) / numTilings
         else
            stateScalingFactor[i] = 1 / numTilings
         end
      end
   end
   local tcOpt = {
      numTilings = numTilings,
      memorySize = memorySize,
      scaleFactor = stateScalingFactor,
      stateMins = stateMins
   }
   local tc = require '../../util/tilecoding'(tcOpt)
   local alphaScaleFactor = opt.alphaScaleFactor or 1/tcOpt.memorySize
   local alpha = alphaScaleFactor / numTilings
   local gamma = opt.gamma
   local lambda = opt.lambda
   local epsilon = opt.epsilon
   local epsilonMinValue = opt.epsilonMinValue
   local epsilonDecayRate = opt.epsilonDecayRate
   local actionSpec = env.getActionSpec()
   local actionType = actionSpec[1]
   local actionMin = actionSpec[3][1]
   local actionMax = actionSpec[3][2]
   local actionList = torch.range(actionSpec[3][1],actionSpec[3][2])
   local nbActions = actionList:size()[1]
   -- Initialize the feature index container
   local featIdx = {}
   -- Initialize weight vector for linear function approximation, arbitrary initialization
   local w = torch.FloatTensor(tcOpt.numTilings * tcOpt.memorySize * nbActions):zero()
   -- Initialize eligibility traces for temporal credit assignment
   local e = torch.FloatTensor(tcOpt.numTilings * tcOpt.memorySize * nbActions):zero()
   function getActionIdx(action)
      local actionIdx
      for i = 1,nbActions do
         if action == actionList[i] then
            actionIdx = i
         end
      end
      return actionIdx
   end
   function getAction(actionIdx)
      return actionList[actionIdx]
   end
   function decayEpsilon()
      -- Decay epsilon on each action selection
      if epsilon > epsilonMinValue then
         epsilon = epsilon * epsilonDecayRate
      end
   end
   function observe(state)
      return tc.feature(state)
   end
   function sumWeights(featIdx)
      local weightSum = 0
      for idx = 1, #featIdx do
         weightSum = weightSum + w[featIdx[idx]]
      end
      return weightSum
   end
   function estimateAllQ(state)
      local Q = torch.Tensor(nbActions):zero()
      for a = 1, nbActions do
         Q[a] = estimateQ(state, getAction(a))
      end
      return Q
   end
   function estimateQ(state, action)
      local featIdx = getFeatures(state, action)
      return sumWeights(featIdx)
   end
   function getFeatures(state, action)
      local obsv = observe(state)
      local actionIdx = getActionIdx(action)
      local featIdx = {}
      for tiling = 1,tcOpt.numTilings do
         featIdx[tiling] = obsv[tiling] + ((actionIdx - 1) * tcOpt.numTilings * tcOpt.memorySize) + 1
      end
      return featIdx
   end
   function accumulateEligibility(featIdx)
      -- accumulate eligibility for all features present in s,a
      for idx = 1, #featIdx do
         e[featIdx[idx]] = e[featIdx[idx]] + 1 -- Accumulating traces
      end
      return e
   end
   function resetEligibility()
      e = e:fill(0)
      return e
   end
   -- Build the agent
   local agent = {}
   function agent.learn(state, action, reward, stateNext, actionNext, terminal)
      if not terminal then
         local _ = accumulateEligibility(getFeatures(state, action))
      end
      local Q = estimateQ(stateNext, actionNext)
      local delta = reward - estimateQ(state, action) + (gamma * Q)
      w = w + (e * alpha * delta)
      -- return the next action for the general TD learning algorithm
      return actionNext
   end
   function agent.selectAction(state)
      local obsv = observe(state)
      local action
      if math.random() < epsilon then
         action = actionList[torch.random(1, nbActions)]
         local _ = resetEligibility()
      else
         local Q = estimateAllQ(state)
         local maxQ, maxIdx = Q:max(1)
         action = getAction(maxIdx[1])
         e = e * gamma * lambda
      end
      if epsilonDecayRate < 1 then
         local _ = decayEpsilon()
      end
      return action
   end
   return agent
end
return getAgent