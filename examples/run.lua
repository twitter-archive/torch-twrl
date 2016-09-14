local params = lapp[[
   -e, --env                  (default "MountainCar-v0")       Gym environment (gym.openai.com/envs)
   -p, --policy               (default "random")     				Gym agent policy (agents/gym/policy)
   -l, --learningUpdate			(default "noLearning")           Gym agent learning update (agents/gym/learn)
   -m, --model            		(default "noModel")              Gym agent model (agents/gym/model)
   -s, --nSteps              	(default 10)                 		Maximum number of steps per episode (envs may define a max steps less than this number)
   -i, --nIterations				(default 2)                 		Number of training iterations
   -v, --video     				(default 200)                 	Record a video every N steps, or false
   -t, --showTrajectory       (default false)                	Show an example learned trajectory on each training iteration
   -r, --renderAllSteps			(default false)                 	Render every step to monitor performance
   -f, --force              	(default false)                 	Force overwrite of the data log
   -u, --resume     				(default false)               	Resume existing experiment
   -p, --timestepsPerBatch    (default 200)              		Number of steps in a training batch (for batch learning update)
   -z, --stepsizeStart			(default 0.3)                  	Starting step size (alpha)
   -g, --gamma           		(default 1)                   	Discount rate
   -o, --policyStd           	(default 0.1)                   	Standard deviation of normal distribution (for normal policy)
   -h, --nHiddenLayerSize		(default 10)                   	Number of hidden layers (for singleHiddenLayer models)
   -g, --gradClip           	(default 5)                   	Maximum absolute value of gradients (for reinforce learning update)
   -b, --baselineType			(default "zeroBaseline")			Type of baseline for advantage calculation ("zeroBaseline" or "padTimeDepAvReturn")
   -w, --weightDecay          (default 0)                   	Optimization parameter: weight decay
   --optimAlpha               (default 0.9)                    Optimization parameter (RMSProp) alpha, smoothing constant
   -a, --beta           		(default 0.01)                   Entropy for forced exploration
   -d, --uploadResults        (default false)                  Automatically upload results to gym leaderboard
   --epsilon                  (default 0.1)                    Epsilon-greedy probability of a random action
   --epsilonDecayRate         (default 0.999)                  Epsilon decay rate
   --epsilonMinValue          (default 0.0001)                 Epsilon minimum value
   --numTiles                 (default 32)                     tilecoding: number of tiles
   --numTilings               (default 4)                      tilecoding: number of overlapping tilings
   --relativeAlpha            (default 0)                      tilecoding: alpha scaling
   --learningType             (default "batch")                Whether the agent learns in 'batch' or 'noBatch'
   --lambda                   (default 0.9)                    Eligibility trace decay
   --initialWeightVal         (default -0.01)                  Initial linear function approximation weights
   --traceType                (default "replacing")            replacing or accumulating traces
   --tdLearnUpdate            (default "qLearning")            qLeanring or SARSA td-learning update
   --actorLearningRate        (default 0.001)                  DDPG - actor network learning rate
   --criticLearningRate       (default 0.0001)                 DDPG - critic network learning rate
   --tau                      (default 0.001)                  DDPG - soft target update parameter
   --bufferSize               (default 10000)                  DDPG - buffer size
   --windowSize               (default 100)                    Performance metric window size
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
params.outdir = logDir .. '/gym'

-- run test
local _ = require 'rl.experiment'(env, agent, nSteps, nIterations, params)





