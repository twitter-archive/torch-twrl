local t  = require 'torch'
local nn = require 'nn'
local os = require 'os'

local function getAgent(opt)
   local opt = opt or {}
   local envDetails = opt.envDetails
   local timestepsPerBatch = opt.timestepsPerBatch or 10
   local learningType = opt.learningType
   local model
   local policy
   local learningUpdate

   opt.nHiddenLayerSize = opt.nHiddenLayerSize or 10
   if opt.model then
      local modelName = opt.model
      model = require('rl.agent.model.' .. opt.model)({
        nInputs = envDetails.nbStates,
        nOutputs = envDetails.nbActions,
        nHiddenLayerSize = opt.nHiddenLayerSize,
        envDetails = envDetails,
        numTilings = opt.numTilings,
        numTiles = opt.numTiles,
        initialWeightVal = opt.initialWeightVal,
        traceType = opt.traceType
      })
   end

   policy = require('rl.agent.policy.' .. opt.policy)({
     client = opt.client,
     instanceID = instanceID,
     nStates = envDetails.nbStates,
     model = model,
     epsilon = opt.epsilon,
     epsilonMinValue = opt.epsilonMinValue,
     epsilonDecayRate = opt.epsilonDecayRate,
     gamma = opt.gamma,
     lambda = opt.lambda,
   })

   local learn = require('rl.agent.learningUpdate.' .. opt.learningUpdate)({
     model = model,
     envDetails = envDetails,
     tdLearnUpdate = opt.tdLearnUpdate,
     gamma = opt.gamma,
     baselineType = opt.baselineType,
     stepsizeStart = opt.stepsizeStart,
     policyStd = opt.policyStd,
     beta = opt.beta,
     gradClip = opt.gradClip,
     weightDecay = opt.weightDecay,
     nIterations = opt.nIterations,
     numTilings = opt.numTilings,
     relativeAlpha = opt.relativeAlpha
   })

   function selectAction(client, instanceID, state)
      local actionSampler = function () return client:env_action_space_sample(instanceID) end
      return policy(state, actionSampler)
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
     local state = (type(opt.state)=='number') and {opt.state} or opt.state
     t.state = torch.DoubleTensor(state)
     local action = (type(opt.action)=='number') and {opt.action} or opt.action
     t.action = torch.DoubleTensor(action)
     t.reward = reward
     local nextState = (type(opt.nextState)=='number') and {opt.nextState} or opt.nextState
     t.nextState = torch.DoubleTensor(opt.nextState)
     local nextAction = (type(opt.nextAction)=='number') and {opt.nextAction} or opt.nextAction
     t.nextAction = torch.DoubleTensor(nextAction)
     t.terminal = (opt.terminal and 1) or 0
     return t
   end
   
   function reward(opt)
      local transition = opt
      -- TODO: decompose batch/iterative for simplicity
      if learningType == 'noBatch' then
        -- iterative learning
        learn(transition, opt.nIter)
      elseif learningType == 'batch' then
        local terminal = opt.terminal
        -- build the transition
        local t = addTrajectory(opt)
        -- add the current transition to the current trajectory
        table.insert(traj, t)
        -- batch learning
        -- on episode end, add full episode trajectory to full list
        if terminal then
           table.insert(trajs, traj)
           timestepsTotal = timestepsTotal + #traj
           traj = {}
        end
        -- learn when we have enough trajectories
        if timestepsTotal >= timestepsPerBatch then
           learn(trajs, opt.nIter)
           resetTrajectories()
        end
      end
   end
   return {
     selectAction = selectAction,
     reward = reward
   }
end
return getAgent