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
   
   local tc = require 'rl.model.tilecoding'({numTilings = numTilings, memorySize = memorySize, scaleFactor = stateScalingFactor, stateMins = stateMins})
   
   local weights = torch.FloatTensor(numTilings * memorySize * nbActions):zero():fill(initialWeightVal)
   local eligibility = torch.FloatTensor(numTilings * memorySize * nbActions):zero():fill(0)

   local function getFeatures(state, action)
      -- get features indecies of Q(s,a) given state and action
      -- featurize the state with the given tilecoder
      local obsv = tc.feature(state)
      local featIdx = {}
      for tiling = 1,numTilings do
         featIdx[tiling] = obsv[tiling] + (action * numTilings * memorySize) + 1
      end
      return featIdx
   end

   -- define accesory methods for the Q function
   local function estimateQ(state, action, w)
      -- sum weights for Q(s,a) for single action
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