local function experiment(envName, agent, nSteps, nIterations, opt)
   local util = require 'rl.util'()
   local gymClient = require '../src/gym-http-api/binding-lua/gym_http_client'
   local opt = opt or {}
   local base = 'http://127.0.0.1:5000'
   local client = gymClient.new(base)
   local instanceID = client:env_create(envName)
   local outdir = opt.outdir
   local video = opt.video
   local showTrajectory = opt.showTrajectory
   local force = opt.force
   local resume = opt.resume
   local renderAllSteps = opt.renderAllSteps

   local perf = require 'rl.perf'({nIterations = nSteps})
   local function run()
      -- Set up the agent given the details about the environment
      client:env_monitor_start(instanceID, outdir, force, resume, video)
      local agentOpt = opt or {}
         agentOpt.stateSpace = client:env_observation_space_info(instanceID)
         agentOpt.actionSpace = client:env_action_space_info(instanceID)
         agentOpt.nIterations = nIterations
         agentOpt.model = agent.model
         agentOpt.policy = agent.policy
         agentOpt.learningUpdate = agent.learningUpdate
         agentOpt.envDetails = util.getStateAndActionSpecs(agentOpt.stateSpace, agentOpt.actionSpace)
      
      local agent = require 'rl.agent.baseAgent'(agentOpt)

      local function actionSampler() return client:env_action_space_sample(instanceID) end

      for nIter = 1,nIterations do
          local state = client:env_reset(instanceID)
          perf.reset()
          for i = 1, nSteps do
             local action = agent.selectAction(client, instanceID, state, envDetails, agent)
             render = renderAllSteps == 'true' and true or false
             nextState, reward, terminal, _ = client:env_step(instanceID, action, render)
             -- terminal if reached max number of steps
             if i == nSteps then terminal = true end
             agent.reward({state = state, reward = reward, terminal = terminal, nextState = nextState, nIter = nIter})
             state = nextState
             perf.addReward(nIter, reward, terminal)
             if terminal then
               state = client:env_reset(instanceID)
            end
          end
          print('Episode: ' .. nIter)
          print(perf.getSummary())
      end
    
      -- Dump result info to disk
      client:env_monitor_close(instanceID)

      if opt.uploadResults == true then
         -- Upload to the scoreboard.
         -- Assumes 'OPENAI_GYM_API_KEY' set on the client side
         -- client:upload() can include algorithm_id and a API key
         client:upload(outdir)
      end
      return true
   end
   if instanceID ~= nil then
      if pcall(run()) then
         print('Error on running experiment!')
      end
   else
      print('Error: No server found! Be sure to start a Gym server before running an experiment.')
   end
end
return experiment
