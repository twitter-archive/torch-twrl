local function experiment(envName, agent, nSteps, nIterations, opt)
   local util = require 'rl.util'()
   local gymClient = require 'rl.binding-lua.gym_http_client'
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
   local render = renderAllSteps == 'true' and true or false
   local perf = require 'rl.perf'({nIterations = nSteps, windowSize = opt.windowSize})
   local function run()
      client:env_monitor_start(instanceID, outdir, force, resume, video)
      local agentOpt = opt or {}
         agentOpt.stateSpace = client:env_observation_space_info(instanceID)
         agentOpt.actionSpace = client:env_action_space_info(instanceID)
         agentOpt.nIterations = nIterations
         agentOpt.model = agent.model
         agentOpt.policy = agent.policy
         agentOpt.learningUpdate = agent.learningUpdate
         agentOpt.envDetails = util.getStateAndActionSpecs(agentOpt.stateSpace, agentOpt.actionSpace)
         function agentOpt.randomActionSampler() return client:env_action_space_sample(instanceID) end
      local agent = require 'rl.agent.baseAgent'(agentOpt)
      local iterPerformance = {}
      for nIter = 1, nIterations do
         collectgarbage()
         perf.reset()
         local state = client:env_reset(instanceID)
         local action = agent.selectAction(client, instanceID, state)
         for i = 1, nSteps do
            nextState, reward, terminal, _ = client:env_step(instanceID, action, render)
            if i == nSteps then terminal = true end
            perf.addReward(nIter, reward, terminal)
            nextAction = agent.selectAction(client, instanceID, nextState)
            agent.reward({state = state, action = action, reward = reward, nextState = nextState, nextAction = nextAction, terminal = terminal, nIter = nIter})
            -- update state and action
            state = nextState
            action = nextAction
            if terminal then break end
         end
         iterPerformance = perf.getSummary(nIter)
      end
    
      -- Dump result info to disk and close the Gym monitor
      client:env_monitor_close(instanceID)

      if opt.uploadResults == 'true' then
         print('Uploading results, check server for URL.')
         -- Upload to the scoreboard, OPENAI_GYM_API_KEY must be set
         client:upload(outdir)
      end
      return iterPerformance
   end
   -- protect the run call to handle errors
   local success, performance
   if instanceID ~= nil then
      success, performance = pcall(run)
      if success == false then
         print(performance)
         print('Error: Experiment was not successfully run.')
         performance = {}
      end
   else
      print('Error: improper configuration. There may be no Gym server started, or your experiment definition may be incomplete.')
      performance = {}
   end
   return performance
end
return experiment