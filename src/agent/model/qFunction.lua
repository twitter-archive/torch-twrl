local function getModel(opt)
   local opt = opt or {}
   local envDetails = opt.envDetails
   local numTilings = opt.numTilings
   local numTiles = opt.numTiles
   local initialWeightVal = opt.initialWeightVal
   local traceType = opt.traceType
   local nbActions = envDetails.nbActions
   local memorySize = numTiles * numTiles
   local tc = require 'twrl.agent.model.tilecoding'({numTilings = numTilings, memorySize = memorySize}) 
   local weights = torch.FloatTensor((numTilings * memorySize) + 1):zero():fill(initialWeightVal)
   local eligibility = torch.FloatTensor((numTilings * memorySize) + 1):zero():fill(0)

   local function getFeatures(state, action)
      local floats = {}
      for i = 1, envDetails.nbStates do
         --TODO: handle when the env space is unbounded
         floats[i] = (numTilings * state[i]) / (envDetails.stateSpec.high[i] - envDetails.stateSpec.low[i])
      end
      if envDetails.actionType == 'Discrete' then
         ints = {action}
      else
         table.insert(floats, action)
      end
      local features = tc.tiles(memorySize, numTilings, floats, ints)
      local featIdx = {}
      for tiling = 1, numTilings do
         featIdx[tiling] = features[tiling] + ((tiling-1) * memorySize) + 1
      end
      -- add a baseline feature
      -- table.insert(featIdx, 1, 1)
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
      -- eligibility = eligibility,
      -- accumulateEligibility = accumulateEligibility,
      estimateQ = estimateQ,
      estimateAllQ = estimateAllQ,
      getFeatures = getFeatures
   }
end
return getModel