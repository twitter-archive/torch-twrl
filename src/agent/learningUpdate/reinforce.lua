local function getLearningUpdate(opt)
   local opt = opt or {}
   local envDetails = opt.envDetails
   local gamma = opt.gamma
   local baselineType = opt.baselineType
   local stepsizeStart = opt.stepsizeStart
   local policyStd = opt.policyStd
   local nIterations = opt.nIterations
   local gradClip = opt.gradClip
   local optim = require 'optim'
   local mo = require 'moses'
   local util = require 'rl.util'()
   local smallEps = 1e-8
   local model = opt.model
   local net = model.net
   local params, gradParams = net:getParameters()
   local gradParamsSq = torch.Tensor():resizeAs(gradParams):fill(0)
   local tmp = torch.Tensor():resizeAs(gradParams)

   local optimConfig = {
    learningRate = -stepsizeStart,
    alpha = opt.optimAlpha,
    weightDecay = opt.weightDecay,
    epsilon = smallEps
   }

	local function learn(trajs, nIter)
      local allTransitions = mo.flatten(trajs, true)
      local numSteps = #allTransitions
      local numEps = #trajs
      local allObservations = torch.DoubleTensor(numSteps, envDetails.nbStates):zero()
      local allActions = torch.DoubleTensor(numSteps, 1):zero()
      for i = 1, numSteps do
         allObservations[i] = allTransitions[i].state
         allActions[i] = allTransitions[i].action
      end

      -- For each set of trajectories calculate compute the discounted sum of rewards
      local trajReturns = {}
      local trajNotTerminals = {}
      local trajLengths = torch.Tensor(numEps):zero()
      for i = 1, numEps do
         trajLengths[i], trajReturns[i], trajNotTerminals[i] = util.discount(trajs[i],gamma)
      end

      -- Compute the baseline
      local baseline = util.getBaseline(trajs, trajLengths, trajReturns, baselineType)

      -- Calculate variance-reduced reward (advantage) function
      local advs = {}
      for i = 1, numEps do
         advs[i] = trajReturns[i] - baseline[{{1,(#trajReturns[i])[1]}}]
      end
      local allAdvantages = torch.DoubleTensor(advs[1])
      for i = 1, numEps-1 do
         allAdvantages = torch.cat(allAdvantages, advs[i+1])
      end
      local N = allObservations:size(1)
      
      -- whiten the advantages to balance negative and positive normalized rewards
      local advantagesNormalized = util.whiten(allAdvantages)

      local function feval(x)
         --reset the gradient parameters
         gradParams:zero() 
         
         -- FORWARD PASS
         local output = net:forward(allObservations)
         
         -- add small constant to avoid nans
         output:add(optimConfig.epsilon)

         -- Define targets for optimization
         --- REINFORCE update for discrete (Reinforce Categorical) and continuous actions (Reinforce Normal)
         local targets = torch.DoubleTensor(N,envDetails.nbActions):zero()
         if envDetails.actionType == 'Discrete' then
            ----------------------------------------
            -- derivative of log categorical w.r.t. p
            -- d ln(f(x,p))     1/p[i]    if i = x
            -- ------------ =
            --     d p          0         otherwise
            ----------------------------------------
            for i = 1, N do
               targets[i][allActions[i][1]+1] = advantagesNormalized[i] * 1/(output[i][allActions[i][1]+1])
            end
         elseif envDetails.actionType == 'Box' then
            ----------------------------------------
            -- Derivative of log normal w.r.t. mean:
            -- d ln(f(x,u,s))   (x - u)
            -- -------------- = -------
            --      d u           s^2
            ----------------------------------------
            for i = 1, N do
               targets[i] = ((output[i] - allActions[i])/(policyStd^2)) * advantagesNormalized[i]
            end
         end
         -- Add gradEntropy to targets to improve exploration and prevent convergence to potentially suboptimal deterministic policy, gradient of entropy of policy (for gradient descent): -(-logp(s) - 1)
         local gradEntropy = torch.log(output) + 1
         targets:add(opt.beta, gradEntropy)
         net:backward(allObservations, targets)
         -- Clip gradients
         if gradClip > 0 then
            gradParams:clamp(-gradClip, gradClip)
         end
         gradParams:div(-1)
         local obj = -torch.mean(torch.sum(targets, 2))
         return obj, gradParams
      end
      
      optimConfig.learningRate = stepsizeStart * ((nIterations - nIter)) / nIterations
      optimConfig.learningRate = mo.max({optimConfig.learningRate, 0.05})
      local params, newObj = optim.rmsprop(feval, params, optimConfig)

      -- Print some learning update details
      print('Learning update at episode: ' .. nIter)
      print('Learning rate: ' .. optimConfig.learningRate)
      print('Number of episodes in learning batch: ' .. numEps)
      print('Number of steps in learning batch: ' .. numSteps)
      end
	return learn
end

return getLearningUpdate
