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