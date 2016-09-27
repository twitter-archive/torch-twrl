params = {
   env = 'CartPole-v0',
   policy = 'categorical',
   learningUpdate = 'reinforce',
   model = 'mlp',
   force = 'true',
   resume = 'false',
   optimAlpha = 0.9,
   timestepsPerBatch = 1000,
   gamma = 1,
   nHiddenLayerSize = 10,
   gradClip = 5,
   baselineType = 'padTimeDepAvReturn',
   beta = 0.01,
   weightDecay = 0,
   windowSize = 10,
   nSteps = 1000,
   nIterations = 1000,
   video = 0,
   optimType = 'rmsprop',
   verboseUpdate = 'false',
   uploadResults = 'false',
   renderAllSteps = 'false',
   learningType = 'batch',
   gymHttpServer = 'http://127.0.0.1:5000'
}

stepsizeStart = {0.3, 0.2, 0.1, 0.01, 0.001}

-- table for the grouped results
results = {}
for i = 1, #stepsizeStart do
   -- Get time, build log folder
   local longDate = os.date("%Y-%m-%dT%H%m%S")
   local uniqueName = longDate .. '-' .. '-' .. params.policy .. '-' .. params.learningUpdate .. '-' .. params.env .. '-stepsizeStart-' .. stepsizeStart[i]
   local logDir = '../../logs/gym/' .. uniqueName
   params.rundir = logDir

   -- environment
   local env = params.env

   -- agent
   local agent = {
      policy = params.policy,
      learningUpdate = params.learningUpdate,
      model = params.model
   }

   -- test details
   local nSteps, nIterations = params.nSteps, params.nIterations

   -- gym data dump directory
   params.outdir = logDir .. '/'

   params.stepsizeStart = stepsizeStart[i]
   local performance = require 'twrl.experiment'(env, agent, nSteps, nIterations, params)
   local stepResults = {}
      stepResults['stepsizeStart'] = params.stepsizeStart
      stepResults['outdir'] = uniqueName
      stepResults['meanEpRewardWindow'] = performance.meanEpRewardWindow
   table.insert(results, stepResults)
end