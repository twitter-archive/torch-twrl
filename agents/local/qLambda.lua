-- QLambda with linear function approximation with tile coding
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
   for i = 1,#stateSpec do
      stateMins[i] = stateSpec[i][3][1]
      stateMaxs[i] = stateSpec[i][3][2]
      if stateMins[i] and stateMaxs[i] then
         stateScalingFactor[i] = (stateMaxs[i] - stateMins[i]) / numTilings
      else
         stateScalingFactor[i] = 1 / numTilings
      end
   end
   local tcOpt = {
      numTilings = numTilings,
      memorySize = memorySize,
      scaleFactor = stateScalingFactor,
      stateMins = stateMins
   }
   print(tcOpt)
   local tc = require '../../util/tilecoding'(tcOpt)
   local alpha = opt.alpha
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
   -- Initialize weight vector for linear function approximation
   -- NOTE: optimize the weights with a negative mean to push exploration, helps convergence
   local w = torch.FloatTensor(tcOpt.numTilings * tcOpt.memorySize * nbActions):zero():fill(-0.01)
   -- Initialize eligibility traces for temporal credit assignment
   local e = torch.FloatTensor(tcOpt.numTilings * tcOpt.memorySize * nbActions):zero()
   -- TODO: action list should be a table, perhaps could use a table_invert to get the associated key
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
      return nil
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
         e[featIdx[idx]] = 1 -- Replacing traces, accumulating would be e[featIdx[idx]] + 1
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
      local delta = 0
      if terminal then
         local _ = resetEligibility()
      else
         local _ = accumulateEligibility(getFeatures(state, action))
         local Q = estimateAllQ(stateNext)
         local maxQ, maxIdx = Q:max(1)
         delta = reward - estimateQ(state, action) + (gamma * maxQ[1])
         w = w + (e * alpha * delta)
      end
      -- can return the next action choice for general TD learning algorithm
      local updatedActionChoice = agent.selectAction(stateNext)
      return updatedActionChoice
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
      local _ = decayEpsilon()
      return action
   end
   return agent
end
return getAgent