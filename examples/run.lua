require 'torch'
require 'pl'
local params = lapp[[
   -b, --base                 (default "gym")                   Use "gym" or "rlenvs"
   -e, --env                  (default "CartPole-v0")           Environment (http://gym.openai.com/envs)
   -p, --policy               (default "categorical")           Agent policy
   -l, --learningUpdate       (default "reinforce")             Agent learning update
   -m, --model                (default "mlp")                   Agent model
   -s, --nSteps               (default 1000)                    Maximum number of steps per episode (envs may define a max steps less than this number)
   -i, --nIterations          (default 1000)                    Number of training iterations
   -v, --video                (default 100)                     Record a video every N steps, or false
   -r, --renderAllSteps       (default false)                   Render every step to monitor performance
   -f, --force                (default true)                    Force overwrite of the data log
   -u, --resume               (default false)                   Resume existing experiment
   -p, --timestepsPerBatch    (default 1000)                    Number of steps in a training batch (for batch learning update)
   -z, --stepsizeStart        (default 0.3)                     Starting step size (alpha)
   -g, --gamma                (default 1)                       Discount rate
   -o, --policyStd            (default 0.1)                     Standard deviation of normal distribution (for normal policy)
   -h, --nHiddenLayerSize     (default 10)                      Number of hidden layers (for singleHiddenLayer models)
   -g, --gradClip             (default 5)                       Maximum absolute value of gradients (for reinforce learning update)
   -b, --baselineType         (default "padTimeDepAvReturn")    Type of baseline for advantage calculation ("zeroBaseline" or "padTimeDepAvReturn")
   -w, --weightDecay          (default 0)                       Optimization parameter: weight decay
   --optimAlpha               (default 0.9)                     Optimization parameter (RMSProp) alpha, smoothing constant
   -a, --beta                 (default 0.01)                    Entropy for forced exploration
   -d, --uploadResults        (default false)                   Automatically upload results to gym leaderboard
   --epsilon                  (default 0.1)                     Epsilon-greedy probability of a random action
   --epsilonDecayRate         (default 0.999)                   Epsilon decay rate
   --epsilonMinValue          (default 0.0001)                  Epsilon minimum value
   --numTiles                 (default 32)                      tilecoding: number of tiles
   --numTilings               (default 4)                       tilecoding: number of overlapping tilings
   --relativeAlpha            (default 0)                       tilecoding: alpha scaling
   --learningType             (default "batch")                 Whether the agent learns in 'batch' or 'noBatch'
   --lambda                   (default 0.9)                     Eligibility trace decay
   --initialWeightVal         (default -0.01)                   Initial linear function approximation weights
   --traceType                (default "replacing")             replacing or accumulating traces
   --tdLearnUpdate            (default "qLearning")             qLeanring or SARSA td-learning update
   --windowSize               (default 10)                      Performance metric window size
   --optimType                (default "rmsprop")               Optimization to use (rmsprop, adam, sgd,...)
   --verboseUpdate            (default true)                    Print details of the learning update
   --gymHttpServer            (default "http://127.0.0.1:5000") Address of Gym Server (https://github.com/openai/gym-http-api)
]]

-- Get time, build log folder
longDate = os.date("%Y-%m-%dT%H:%m:%S.000")
logDir = '../../logs/gym/' .. longDate .. '-' .. '-' .. params.policy .. '-' .. params.learningUpdate .. '-' .. params.env
params.rundir = logDir
paths.mkdir(logDir)

-- create log file
cmd = torch.CmdLine()
cmd:log(logDir .. '/log', params)

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

-- run test
local performance = require 'twrl.experiment'(env, agent, nSteps, nIterations, params)
