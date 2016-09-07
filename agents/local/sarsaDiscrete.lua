-- this is a tabular on-policy SARSA for TD Control
-- it uses epsilon-greedy Q action selectioon
local function getAgent(opt)
   local opt = opt or {}
   local env = opt.env or nil
   local alpha = opt.alpha
   local gamma = opt.gamma
   local epsilon = opt.epsilon
   local nStates = env.getStateSpec()
   local nActions = env.getActionSpec()[3][2]
   local actionMin = env.getActionSpec()[3][1]
   local actionMax = env.getActionSpec()[3][2]
   local nDims = #nStates
   local maxDims = {}
   local nStateSpace = 1
   -- if this is single dimensional
   if nStates[1][1] == nil then
      nStateSpace = nStateSpace * nStates[3][2]
   else
      -- if this is multidimensional
      for d = 1,nDims do
         nStateSpace = nStateSpace * nStates[d][3][2]
      end
   end
   local Q = torch.Tensor(nStateSpace,nActions):zero()
   local agent = {}
   local function sub2ind(state)
      -- convert subscript to index in 1d vector given matrix size
      -- if state space is defined by a MxN matrix
      local m = nStates[1][3][2]
      local r = state[1]
      local c = state[2]
      local index = ((c-1)*m)+r
      return index
   end
   local function observe(state)
      -- hashing takes in environment and returns agent state
      local o = 0
      -- if the state space is defined by a single int dimension
      if nStates[1][1] == nil then
         o = state
      elseif nDims == 2 then
         -- if the state space is defined by two int dimensions
         o = sub2ind(state)
      end
      return o
   end
   function agent.learn(state, action, reward, stateNext, actionNext, terminal)
      local o = observe(state)
      local o_next = observe(stateNext)
      local Qnow = Q[{o, action}]
      local Qnext = Q[{o_next, actionNext}]
      local tdError = reward + gamma * Qnext - Qnow
      Q[{o, action}] = Qnow + alpha*tdError
      return actionNext
   end
   function agent.selectAction(state)
      local o = observe(state)
      local action = 0
      -- Epsilon-greedy policy
      if torch.uniform() < epsilon then
         action = torch.random(actionMin,actionMax)
      else
         -- Get the maximizing action
         -- print(Q, Q[o])
         local _, maxIdx = Q[o]:max(1)
         action = maxIdx[1]
      end
      -- make sure you are returning a discrete action
      return math.floor(action)
   end
   return agent
end
return getAgent