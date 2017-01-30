local function experiment(envName, agent, nSteps, nIterations, opt)
   local util = require 'twrl.util'()
   local opt = opt or {}
   local client, instanceID
   if opt.base == 'gym' then
       local gymClient = require 'twrl.binding-lua.gym_http_client'
       gymHttpServer = opt.gymHttpServer or 'http://127.0.0.1:5000'
       client = gymClient.new(gymHttpServer)
       instanceID = client:env_create(envName)
   else
       local rlenvsClient = require 'twrl.client'
       client = rlenvsClient.new()
       instanceID = client:env_create(envName, opt)
   end
   local outdir = opt.outdir
   local video = opt.video
   local force = opt.force
   local resume = opt.resume
   local renderAllSteps = opt.renderAllSteps
   local render = renderAllSteps == 'true' and true or false
   local perf = require 'twrl.perf'({nIterations = nSteps, windowSize = opt.windowSize})
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
      local agent = require 'twrl.agent.baseAgent'(agentOpt)
      local iterPerformance = {}

      for nIter = 1, nIterations do
         perf.reset()
         local state = client:env_reset(instanceID)
         local action = agent.selectAction(client, instanceID, state)
         for i = 1, nSteps do
            local nextState, reward, terminal, _ = client:env_step(instanceID, action, render)
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
         print('Uploading results, check server for URL')
         -- Upload to the scoreboard, OPENAI_GYM_API_KEY must be set
         client:upload(outdir)
      end
      return iterPerformance
   end
   -- protect the run call to handle errors
   local success, performance
   function errorHandle(error)
      print('Error: Experiment was not successfully run.')
      print(error)
      print(debug.traceback())
      return {}
   end
   if instanceID ~= nil then
      success, performance, ret = xpcall(run, errorHandle)
      if success == false then
         performance = {}
      end
   else
      print('Error: improper configuration. There may be no Gym server started, or your experiment definition may be incomplete.')
      print('instanceID is nil')
      performance = {}
   end
   return performance
end
return experiment
