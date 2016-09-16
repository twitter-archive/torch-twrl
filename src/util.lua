local function utilities()
   local util = {}
   function util.getStateAndActionSpecs(stateSpec, actionSpec)
      -- Get environment specifications in case they are needed
      local nbStates, nbStatesDiscrete
      if stateSpec['name'] == 'Discrete' then
         nbStates = 1
         nbStatesDiscrete = stateSpec['n']
      else
         totalStateSpace = 1
         for i = 1, #stateSpec['shape'] do
            totalStateSpace = totalStateSpace * stateSpec['shape'][i]
         end
         nbStates = totalStateSpace
      end
      local nbActions, nbActionSpace = 0
      local actionType
      local actionSpaceBounds
      if actionSpec['name'] == 'Discrete' then
         nbActions = actionSpec['n']
         actionType = 'Discrete'
      else
         nbActionSpace = actionSpec['shape'][1]
         actionType = 'Box'
         actionSpaceBounds = torch.Tensor(nbActionSpace, 3):zero()
         for i = 1, nbActionSpace do
            actionSpaceBounds[i][1] = actionSpec['low'][i]
            actionSpaceBounds[i][2] = actionSpec['high'][i]
            actionSpaceBounds[i][3] = actionSpaceBounds[i][2] - actionSpaceBounds[i][1]
         end
         nbActions = nbActionSpace
      end
      envDetails = {
         stateSpec = stateSpec,
         nbStates = nbStates,
         nbStatesDiscrete = nbStatesDiscrete,
         totalStateSpace = totalStateSpace,
         actionSpec = actionSpec,
         actionType = actionType,
         actionSpaceBounds = actionSpaceBounds,
         nbActionSpace = nbActionSpace,
         nbActions = nbActions
      }
      return envDetails
   end
   return util
end
return utilities