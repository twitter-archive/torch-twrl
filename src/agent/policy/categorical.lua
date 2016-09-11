local function getPolicy(opt)
   local opt = opt or {}
   local client = opt.client
   local instanceID = instanceID
   local nStates = opt.nStates
   local model = opt.model

   local function selectAction(state, actionSampler)
      -- autocast state to a table, to handle cast to tensor
      local state = (type(state) == 'number') and {state} or state
      local obsv = torch.DoubleTensor(state):reshape(1,nStates)
      local out = model:forward(obsv)
      local action
      -- Single discrete action space, action selection
      -- based on the sampling of the, softmax probabilities output by the network
      if out:ne(out):sum() > 0 then
         print('Error in action selection, selecting randomly')
         print(obsv, out, out:ne(out))
         action = actionSampler()
      else
         -- Sample action ~ p(s; θ)
         action = (torch.multinomial(out, 1)-1)[1][1]
      end
      return action
   end
   return selectAction
end
return getPolicy
