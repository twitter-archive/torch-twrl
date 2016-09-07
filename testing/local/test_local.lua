local params = lapp[[
   -e, --env                  (default "RandomWalk")        Local environment (/envs)
   -a, --agent                (default "sarsaDiscrete")     Local agent (agents/local)
   -s, --nSteps               (default 100)                 Max steps per episode
   -p, --nEpisodes            (default 1000)                Number of episodes
   -r, --nReport              (default 100)                 Results every N episodes
   -a, --alpha                (default 0.2)                 Learning rate
   -f, --alphaScaleFactor     (default 0.5)                 tilecoding: alpha scaling
   -g, --gamma                (default 0.99)                Discount rate
   -l, --lambda               (default 0.9)                 Eligibility trace decay
   -n, --epsilon              (default 1.0)                 Greedy action selection
   -d, --epsilonDecayRate     (default 0.999)               Epsilon decay rate
   -m, --epsilonMinValue      (default 0.0001)              Epsilon minimum value
   -t, --numTiles             (default 32)                  tilecoding: number of tiles
   -g, --numTilings           (default 4)                   tilecoding: number of overlapping tilings
]]

-- Get time, build log folder
longDate = os.date("%Y-%m-%dT%H:%m:%S.000")
logDir = '../../logs/local/' .. longDate .. '-' .. '-' .. params.agent .. '-' .. params.env
params.rundir = logDir
paths.mkdir(logDir)

-- create log file
cmd = torch.CmdLine()
cmd:log(logDir .. '/log', params)

-- environment
local envStr = '../../envs/' .. params.env
local env = require(envStr)()

-- agent
local agentOpt = params
agentOpt.env = env
local agentStr = '../../agents/local/' .. params.agent
local agent = require(agentStr)(agentOpt)

-- test details
local nSteps, nEpisodes, nReport = params.nSteps, params.nEpisodes, params.nReport

-- run test
local _ = require 'experiment_local'(env, agent, nSteps, nEpisodes, nReport)