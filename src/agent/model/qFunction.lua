local function getModel(opt)
   local opt = opt or {}
   local envDetails = opt.envDetails
   local numTilings = opt.numTilings
   local numTiles = opt.numTiles
   local initialWeightVal = opt.initialWeightVal
   local traceType = opt.traceType
   local nbActions = envDetails.nbActions

   local function getStateMinsAndScaling(envDetails, numTilings)
      local nbStates = envDetails.nbStates
      local stateMins = envDetails.stateSpec.low
      local stateScalingFactor = {}
      for i = 1, nbStates do
         -- ensure that the environment limits are not infinite for scaling
         if (envDetails.stateSpec.low[i] and envDetails.stateSpec.high[i]) and (envDetails.stateSpec.high[i] < 1000) then
            stateScalingFactor[i] = (envDetails.stateSpec.high[i] - envDetails.stateSpec.low[i]) / numTilings
         else
            stateScalingFactor[i] = 1/numTilings
         end
      end
      return stateScalingFactor, stateMins
   end

   local stateScalingFactor, stateMins = getStateMinsAndScaling(envDetails, numTilings)
   local memorySize = numTiles * numTiles
   local tc = require 'twrl.agent.model.tilecoding'({numTilings = numTilings, memorySize = memorySize}) 
   local weights = torch.FloatTensor((numTilings * memorySize) + 1):zero():fill(initialWeightVal)
   local eligibility = torch.FloatTensor((numTilings * memorySize) + 1):zero():fill(0)

   local function getFeatures(state, action)
      -- TODO: fix the box actions to append to floats
      floats = state
      if envDetails.actionType == 'Discrete' then
         ints = {action}
      else
         table.insert(floats, action)
      end
      features = tc.tiles(memorySize, numTilings, floats, ints)
      featIdx = {}
      for tiling = 1, numTilings do
         featIdx[tiling] = features[tiling] + ((tiling-1) * memorySize) + 1
      end
      -- add a baseline feature
      table.insert(featIdx, 1, 1)
      return featIdx
   end

   -- define accesory methods for the Q function
   local function estimateQ(state, action, w)
      -- estimate Q(s,a) as sum of corresponding weights
      local featIdx = getFeatures(state, action)
      local weightSum = 0
      for idx = 1, #featIdx do
         weightSum = weightSum + w[featIdx[idx]]
      end
      return weightSum
   end

   local function estimateAllQ(state, w)
      -- estimate Q(s,a) for all actions
      local qVals = torch.Tensor(nbActions):zero()
      -- actions are base 0 for environment simplicity
      for action = 0, nbActions-1 do
         qVals[action+1] = estimateQ(state, action, w)
      end
      return qVals
   end
   
   local function accumulateEligibility(state, action, elig)
      local featIdx = getFeatures(state, action)
      -- accumulate eligibility for all features present in s,a
      for idx = 1, #featIdx do
         if traceType == 'replacing' then
            elig[featIdx[idx]] = 1
         elseif traceType == 'accumulating' then
            elig[featIdx[idx]] = elig[featIdx[idx]] + 1
         end
         eligibility = elig
      end
   end

   return { 
      weights = weights,
      eligibility = eligibility,
      accumulateEligibility = accumulateEligibility,
      estimateQ = estimateQ,
      estimateAllQ = estimateAllQ,
   }
end
return getModel