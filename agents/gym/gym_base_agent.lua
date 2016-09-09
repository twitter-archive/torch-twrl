-- gym-reinforce-new.lua
-- vanilla policy gradient
local t  = require 'torch'
local nn = require 'nn'
local os = require 'os'
local tj = require '../../util/trajectory'()
local gum = require '../../util/gym_utilities'()

local function getAgent(opt)
   local opt = opt or {}
   local envDetails = opt.envDetails
   local timestepsPerBatch = opt.timestepsPerBatch or 10

   local latestAction
   local latestState
   local previousAction
   local previousState

   local model
   local policy
   local learningUpdate

   opt.nHiddenLayerSize = opt.nHiddenLayerSize or 10
   if opt.model then
      local modelName = opt.model
      model = require('../gym/model/' .. 'mlp')({
        nInputs = envDetails.nbStates,
        nOutputs = envDetails.nbActions,
        nHiddenLayerSize = opt.nHiddenLayerSize}
      )
      -- TODO: fix how the agents with model parameters are identified
      --[[ these should be wrapped in the model itself
      if (modelName == 'singleHiddenLayerCategorical') or (modelName == 'singleHiddenLayerNormal') then
         agent.theta, agent.gradTheta = agent.model:getParameters()
         agent.gradThetaSq = torch.Tensor(agent.gradTheta:size()):zero()
      elseif (modelName == 'ddpgActorCritic') then
         agent.theta, agent.gradTheta = agent.model.actor:getParameters()
         agent.gradThetaSq = torch.Tensor(agent.gradTheta:size()):zero()
      end
      ]]
      print('Model: ' .. modelName)
   end

   policy = require('../gym/policy/' .. opt.policy)({
     client = client,
     instance_id = instance_id,
     nStates = envDetails.nbStates,
     model = model.model
   })
   local learn = require('../gym/learningUpdate/' .. opt.learningUpdate)({
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
   --[[ this calculation should be done in the learningUpdate
   -- calculate the alpha to start with
   if agent.alphaScaleFactor ~= 0 then
      -- relative step size if given a scale factor and number of tilings
      agent.alpha = agent.alphaScaleFactor / agent.numTilings
   else
      -- set alpha to the starting step size when given
      agent.alpha = agent.stepsizeStart
   end
   ]]

   function selectAction(client, instance_id, state)
      local action = policy(state)
      previousAction = latestAction
      latestAction = action
      return action
   end

   local timestepsTotal = 0
   local trajCount = 1
   local trajs = {}
   local traj = {}

   function resetTrajectories()
     local _ = tj.clearTrajs()
     timestepsTotal = 0
     trajCount = 1
     trajs = {}
     traj = {}
   end

   function addTrajectory(opt)
     local t = {}
     state = (type(opt.state)=='number') and {opt.state} or opt.state
     t.state = torch.DoubleTensor(state)
     action = (type(opt.action)=='number') and {opt.action} or action
     t.action = torch.DoubleTensor(action)
     t.reward = reward
     t.nextState = torch.DoubleTensor(nextState)
     t.terminal = (terminal and 1) or 0
     --tj.pushTraj(t)
     return t
   end

   function reward(opt)
      local terminal = opt.terminal
      opt.action = latestAction
      local t = addTrajectory(opt)
      table.insert(traj, t)
      if terminal then
         timestepsTotal = timestepsTotal + #traj
         table.insert(trajs, traj)
         tj.pushTraj(traj)
         traj = {}
      end
      if timestepsTotal >= timestepsPerBatch then
        learn(trajs, tj)
        resetTrajectories()
      end
   end

   --[[
   function agent.collectTrajectories(client, instance_id, nSteps, render)
      local _ = tj.clearTrajs()
      local timestepsTotal = 0
      local trajCount = 1
      local render = render
      local trajs = {}
      while timestepsTotal < agent.timestepsPerBatch do
         trajs[trajCount] = agent.getTrajectory(client, instance_id, nSteps, render)
         local _ = tj.pushTraj(trajs[trajCount])
         timestepsTotal = timestepsTotal + #trajs[trajCount]
         trajCount = trajCount + 1
      end
      local episodeRewards = torch.Tensor(#trajs):zero()
      local episodeLengths = torch.Tensor(#trajs):zero()
      for i = 1,#trajs do
         episodeLengths[i] = #trajs[i]
         for j = 1,#trajs[i] do
            episodeRewards[i] = episodeRewards[i] + trajs[i][j].reward
         end
      end
      return trajs, timestepsTotal, episodeLengths, episodeRewards, tj
   end
   function agent.getTrajectory(client, instance_id, nSteps, render, learningType)
      -- Run the agent for an episode (trajectory through the environment)
      local render = render

      local traj = {}
      local nextAction
      local updatedActionChoice = nil

      -- get the first state and action
      local state = client:env_reset(instance_id)
      local action = agent.selectAction(client, instance_id, state, envDetails, agent)

      for i = 1, nSteps do
         local nextState, reward, terminal
         -- TODO: clean up this if statement
         if render == 'false' then
            nextState, reward, terminal, _ = client:env_step(instance_id, action)
         else
            nextState, reward, terminal, _ = client:env_step(instance_id, action, render)
         end

         -- set terminal to true if reached max number of steps
         if i == nSteps then
            terminal = true
         end

         -- select the next action
         nextAction = agent.selectAction(client, instance_id, state, envDetails, agent)
         if learningType == 'noBatch' then
            -- Perform the learning update with the selected next action if needed
            local _ = agent.learn(state, action, reward, nextState, nextAction, terminal, agent)
            -- pass through the updated action AFTER learning update if algorithm demands
            updatedActionChoice = agent.selectAction(client, instance_id, nextState, envDetails, agent)
         end

         -- Store the step to a trajectory table
         -- autocast tables, to handle cast to tensor
         traj[i] = {}
         state = (type(state)=='number') and {state} or state
         traj[i].state = torch.DoubleTensor(state)
         action = (type(action)=='number') and {action} or action
         traj[i].action = torch.DoubleTensor(action)
         traj[i].reward = reward
         traj[i].nextState = torch.DoubleTensor(nextState)
         traj[i].terminal = (terminal and 1) or 0

         -- set the state to the next state
         state = nextState

         -- set the action to the next action
         if updatedActionChoice ~= nil then
            action = updatedActionChoice
         else
            action = nextAction
         end

         -- break if terminal
         if terminal then break end
      end
      return traj
   end
   ]]
   return {
     selectAction = selectAction,
     reward = reward
   }
end
return getAgent
