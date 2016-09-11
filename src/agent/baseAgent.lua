local t  = require 'torch'
local nn = require 'nn'
local os = require 'os'
-- local tj = require 'rl.trajectory'()

local function getAgent(opt)
   local opt = opt or {}
   local envDetails = opt.envDetails
   local timestepsPerBatch = opt.timestepsPerBatch or 10

   -- local latestAction
   -- local latestState
   -- local previousAction
   -- local previousState

   local model
   local policy
   local learningUpdate

   opt.nHiddenLayerSize = opt.nHiddenLayerSize or 10
   if opt.model then
      local modelName = opt.model
      model = require('rl.agent.model.mlp')({
        nInputs = envDetails.nbStates,
        nOutputs = envDetails.nbActions,
        nHiddenLayerSize = opt.nHiddenLayerSize}
      )
      print('Model: ' .. modelName)
   end

   policy = require('rl.agent.policy.' .. opt.policy)({
     client = opt.client,
     instanceID = instanceID,
     nStates = envDetails.nbStates,
     model = model.model
   })

   local learn = require('rl.agent.learningUpdate.' .. opt.learningUpdate)({
     model = model,
     envDetails = envDetails,
     gamma = opt.gamma,
     baselineType = opt.baselineType,
     stepsizeStart = opt.stepsizeStart,
     policyStd = opt.policyStd,
     beta = opt.beta,
     gradClip = opt.gradClip,
     weightDecay = opt.weightDecay,
     nIterations = opt.nIterations
   })

   function selectAction(client, instanceID, state)
      local actionSampler = function () return client:env_action_space_sample(instanceID) end
      local action = policy(state, actionSampler)
      -- previousAction = latestAction
      -- latestAction = action
      return action
   end

   local timestepsTotal = 0
   local trajs = {}
   local traj = {}

   function resetTrajectories()
     timestepsTotal = 0
     trajs = {}
     traj = {}
   end

   function addTrajectory(opt)
     local t = {}
     state = (type(opt.state)=='number') and {opt.state} or opt.state
     t.state = torch.DoubleTensor(state)
     action = (type(opt.action)=='number') and {opt.action} or opt.action
     t.action = torch.DoubleTensor(action)
     t.reward = reward
     t.nextState = torch.DoubleTensor(opt.nextState)
     t.terminal = (opt.terminal and 1) or 0
     return t
   end
   
   function reward(opt)
      local terminal = opt.terminal
      -- build the transition
      local t = addTrajectory(opt)
      -- add the current transition to the current trajectory
      table.insert(traj, t)
      if terminal then
         -- episode is over, add current trajectory to full list 
         table.insert(trajs, traj)
         timestepsTotal = timestepsTotal + #traj
         -- clear the episode trajectory table
         traj = {}
      end
      -- learn when we have enough trajectories
      if timestepsTotal >= timestepsPerBatch then
         learn(trajs, opt.nIter)
         resetTrajectories()
      end
   end
   return {
     selectAction = selectAction,
     reward = reward
   }
end
return getAgent
