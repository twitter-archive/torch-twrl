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

   -- Build the agent with the options
   local agent = opt

   if opt.model then
      local modelName = opt.model
      agent.model = require('../gym/model/' .. opt.model)(envDetails.nbStates, envDetails.nbActions, agent)
      -- TODO: fix how the agents wit h model parameters are identified
      if (modelName == 'singleHiddenLayerCategorical') or (modelName == 'singleHiddenLayerNormal') then
         agent.theta, agent.gradTheta = agent.model:getParameters()
         agent.gradThetaSq = torch.Tensor(agent.gradTheta:size()):zero()
      elseif (modelName == 'ddpgActorCritic') then
         print(agent.model.actor)
         print(agent.model.critic)
         -- Initialize the last layer parameters
         agent.model.actor.modules[7].weight:uniform(-3e-4, 3e-4)
         agent.model.actor.modules[7].bias:zero()
         agent.model.critic.modules[6].weight:uniform(-3e-4, 3e-4)
         agent.model.critic.modules[6].bias:zero()
         -- build the target networks
         agent.model.actor_target = agent.model.actor:clone('running_mean', 'running_std')
         agent.model.critic_target = agent.model.critic:clone('running_mean', 'running_std')

         -- get the parameters of the actor/critics and target networks
         -- agent.theta, agent.gradTheta = agent.model.actor:getParameters()
         -- agent.gradThetaSq = torch.Tensor(agent.gradTheta:size()):zero()
         agent.model.params = {}
         agent.model.gradParams = {}
         agent.model.params.actor, agent.model.gradParams.actor = agent.model.actor:getParameters()
         agent.model.params.actor_target, _ = agent.model.actor_target:getParameters()
         agent.model.params.critic, agent.model.gradParams.critic = agent.model.critic:getParameters()
         agent.model.params.critic_target, _ = agent.model.critic_target:getParameters()
         
         -- Set criterion
         criterion = nn.MSECriterion()
         -- set the crtiterion for the opmization
      end
      print('Model: ' .. modelName)
   end
   agent.selectAction = require('../gym/policy/' .. opt.policy)
   agent.learn = require('../gym/learningUpdate/' .. opt.learningUpdate)
   
   -- calculate the alpha to start with
   if agent.alphaScaleFactor ~= 0 then
      -- relative step size if given a scale factor and number of tilings
      agent.alpha = agent.alphaScaleFactor / agent.numTilings
   else
      -- set alpha to the starting step size when given
      agent.alpha = agent.stepsizeStart
   end

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
         nextAction = agent.selectAction(client, instance_id, nextState, envDetails, agent)
         
         if learningType == 'noBatch' then
            -- Perform the learning update with the selected next action if needed
            local _ = agent.learn(state, action, reward, nextState, nextAction, terminal, agent)
            -- pass through the updated action AFTER learning update if algorithm demands
            -- updatedActionChoice = agent.selectAction(client, instance_id, nextState, envDetails, agent)
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
         action = nextAction

         -- -- set the action to the next action
         -- if updatedActionChoice ~= nil then
         --    action = updatedActionChoice
         -- else
         --    action = nextAction
         -- end
         -- break if terminal

         if terminal then break end
      end
      return traj
   end
   return agent
end
return getAgent