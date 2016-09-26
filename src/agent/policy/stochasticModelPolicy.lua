local function getStochasticModelPolicy(opt)
   -- function to sample actions based on the model
   local actionSampler = opt.actionSampler

   local function getPolicy(opt)
      local opt = opt or {}
      local client = opt.client
      local instanceID = instanceID
      local envDetails = opt.envDetails
      local nStates = envDetails.nbStates
      local model = opt.model.net

      if envDetails.actionType == 'Discrete' then
         opt.actionShift = 1
      elseif envDetails.actionType == 'Box' then
         local actionBoundFactor = torch.Tensor(envDetails.nbActionSpace):zero()
         for i = 1, envDetails.nbActionSpace do
            actionBoundFactor[i] =  envDetails.actionSpaceBounds[i][3] / 2.0
         end
         opt.actionBoundFactor = actionBoundFactor
      end

      local function selectAction(state)
         -- autocast state to a table, to handle cast to tensor
         local state = (type(state) == 'number') and {state} or state
         local obsv = torch.DoubleTensor(state):reshape(1,envDetails.nbStates)
         local out = model:forward(obsv)
         return actionSampler(out, opt)
      end
      return selectAction
   end

   return getPolicy
end

return getStochasticModelPolicy