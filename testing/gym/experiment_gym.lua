local GymClient = require('../../util/gym-http-api/binding-lua/gym_http_client')
local gum = require '../../util/gym_utilities'()
local function testGym(envName, agent, nSteps, nIterations, opt)
   local opt = opt or {}
   local base = 'http://127.0.0.1:5000'
   local client = GymClient.new(base)
   local instance_id = client:env_create(envName)
   local outdir = opt.outdir
   local video = opt.video
   local showTrajectory = opt.showTrajectory
   local force = opt.force
   local resume = opt.resume
   local renderAllSteps = opt.renderAllSteps

   local function run()
      -- Set up the agent given the details about the environment
      client:env_monitor_start(instance_id, outdir, force, resume, video)
      local agentOpt = opt or {}
         agentOpt.stateSpace = client:env_observation_space_info(instance_id)
         agentOpt.actionSpace = client:env_action_space_info(instance_id)
         agentOpt.nIterations = nIterations
         agentOpt.model = agent.model
         agentOpt.policy = agent.policy
         agentOpt.learningUpdate = agent.learningUpdate
         agentOpt.envDetails = gum.getStateAndActionSpecs(agentOpt.stateSpace, agentOpt.actionSpace)
      local agent = require('../../agents/gym/gym_base_agent')(agentOpt)

      for j=1,nIterations do
            local state = client:env_reset(instance_id)
            local action = agent.selectAction(client, instance_id, state, envDetails, agent)
            for i = 1, nSteps do
               local nextState, reward, terminal
               -- TODO: clean up this if statement
               render = render == 'true' and true or false
               nextState, reward, terminal, _ = client:env_step(instance_id, action, render)

               -- set terminal to true if reached max number of steps
               if i == nSteps then terminal = true end

               -- select the next action
               --nextAction = agent.selectAction(client, instance_id, state, envDetails, agent)
               agent.reward({state = state, action = action, reward = reward, terminal = terminal, agent = agent})
               action = agent.selectAction(client, instance_id, state, envDetails, agent)
               --[[if learningType == 'noBatch' then
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
               ]]

               if terminal then break end
            end


      if agentOpt.learningType == 'noBatch' then
         local trajs = {}
         local episodeRewards = torch.Tensor(nIterations):zero()
         for nIter = 1, nIterations do
            trajs[nIter] = agent.getTrajectory(client, instance_id, nSteps, renderAllSteps, agentOpt.learningType)
            for j = 1,#trajs[nIter] do
               episodeRewards[nIter] = episodeRewards[nIter] + trajs[nIter][j].reward
            end
            print('------------------')
            print('Episode: ' .. nIter)
            print('Steps: ' .. #trajs[nIter])
            print('Reward: ' .. episodeRewards[nIter])
            print('------------------')
         end
      else
         -- run the learning algorithm over a number of iterations
         for nIter = 1, nIterations do
            local trajs, timestepsTotal, epLens, epRews, tj = agent.collectTrajectories(client, instance_id, nSteps, renderAllSteps)
            local _ = agent.learn(trajs, nIter, agentOpt.envDetails, tj, agent)
            print('------------------')
            print('Iteration: ' .. nIter)
            print('NumTraj: ' .. #trajs)
            print('NumTimesteps: ' .. timestepsTotal)
            print('MaxRew: ' .. epRews:max())
            print('MeanRew: ' .. epRews:mean())
            print('MeanLen: ' .. epLens:mean())
            print "-----------------"
            if showTrajectory then
               agent.getTrajectory(client, instance_id, nSteps, showTrajectory, agentOpt.learningType)
            end
         end
      end

      -- Dump result info to disk
      client:env_monitor_close(instance_id)

      if opt.uploadResults == true then
         -- Upload to the scoreboard.
         -- Assumes 'OPENAI_GYM_API_KEY' set on the client side
         -- client:upload() can include algorithm_id and a API key
         client:upload(outdir)
      end
      return true
   end
   if instance_id ~= nil then
      if pcall(run()) then
         print('Error on running experiment!')
      end
   else
      print('Error: No server found! Be sure to start a Gym server before running an experiment.')
   end
end
return testGym
