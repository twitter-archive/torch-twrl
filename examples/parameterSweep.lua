-- parameterSweep

params = {
	env = 'CartPole-v0',
	policy = 'categorical',
	learningUpdate = 'reinforce',
	model = 'mlp',
	nIterations = 50,
	nSteps = 1000,
	windowSize = 10
}

-- Get time, build log folder
longDate = os.date("%Y-%m-%dT%H:%m:%S.000")
logDir = '../../logs/gym/' .. longDate .. '-' .. '-' .. params.policy .. '-' .. params.learningUpdate .. '-' .. params.env
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

stepsizeStart = {0.1, 0.01, 0.001}

results = {}
for i = 1,10 do
	-- random search over parameters
	params.stepsizeStart = stepsizeStart[math.random(#stepsizeStart)]

	-- run test
	local performance = require 'rl.experiment'(env, agent, nSteps, nIterations, params)
	print({i, params.stepsizeStart, performance.meanEpRewardWindow})
	table.insert(results, {i, params.stepsizeStart, performance.meanEpRewardWindow})
end

print(results)
