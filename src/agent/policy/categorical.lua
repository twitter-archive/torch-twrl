local function getPolicy(opt)
   local opt = opt or {}
   local client = opt.client
   local instanceID = instanceID
   local nStates = opt.nStates
   local model = opt.model.model

   local function selectAction(state, actionSampler)
      -- autocast state to a table, to handle cast to tensor
      local state = (type(state) == 'number') and {state} or state
      local obsv = torch.DoubleTensor(state):reshape(1,nStates)
      local out = model:forward(obsv)
      out = torch.exp(out)
      -- Sample action from distribution ~ p(s; θ) 
      return (torch.multinomial(out, 1)-1)[1][1]
   end
   return selectAction
end
return getPolicy
