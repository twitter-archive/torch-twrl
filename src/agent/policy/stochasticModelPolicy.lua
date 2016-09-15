local function getStochasticModelPolicy(opt)
   -- function to sample actions based on the model
   local actionSampler = opt.actionSampler

   local function getPolicy(opt)
      local opt = opt or {}
      local client = opt.client
      local instanceID = instanceID
      local nStates = opt.nStates
      local model = opt.model.net

      local function selectAction(state)
         -- autocast state to a table, to handle cast to tensor
         local state = (type(state) == 'number') and {state} or state
         local obsv = torch.DoubleTensor(state):reshape(1,nStates)
         local out = model:forward(obsv)
         return actionSampler(out, opt)
      end
      return selectAction
   end

   return getPolicy
end

return getStochasticModelPolicy