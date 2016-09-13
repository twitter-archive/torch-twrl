local function getLearningUpdate(opt)
   local opt = opt or {}
   local modelP = opt.model
   local model = modelP.model
   local envDetails = opt.envDetails
   local gamma = opt.gamma
   local baselineType = opt.baselineType
   local stepsizeStart = opt.stepsizeStart
   local policyStd = opt.policyStd
   local nIterations = opt.nIterations
   local optim = require 'optim'
   local mo = require 'moses'
   local util = require 'rl.util'()

   local config = {
    learningRate = stepsizeStart,
    weightDecay = opt.weightDecay
   }

   local criterion = nn.MSECriterion()

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

      local function feval(x)
        local output = model:forward(allObservations)

        -- whiten the advantages to balance rewards
        local advantagesNormalized = util.whiten(allAdvantages)

        -- decrement the step size, helps with convergence
        stepsize = stepsizeStart * ((nIterations - nIter) / nIterations)

        -- Calculate (negative of) gradient of entropy of policy (for gradient descent): -(-logp(s) - 1)
        -- add small constant to avoid nans
        output:add(1e-10)
        -- local gradEntropy = torch.log(output) + 1

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

         -- Add to gradEntropy to targets to improve exploration and prevent convergence to potentially suboptimal deterministic policy
         -- targets:add(opt.beta, gradEntropy)
         maxVal = 1
         local obj = -(torch.mean(targets,2)/maxVal)
         -- criterion:forward(output, targets)
         -- local err = criterion:backward(output, targets)
         model:backward(allObservations, targets)
         return obj, modelP.gradTheta
      end

      -- TODO: change to params
      optim.rmsprop(feval, modelP.theta, config)

       -- modelP.gradThetaSq = modelP.gradThetaSq * opt.weightDecay + torch.pow(modelP.gradTheta, 2) * (1 - opt.weightDecay)
       -- if opt.gradClip > 0 then
       --   modelP.gradTheta:clamp(-opt.gradClip, opt.gradClip)
       -- end

       -- modelP.theta:add(torch.cdiv(modelP.gradTheta * stepsize, torch.sqrt(modelP.gradThetaSq) + 1e-10))


       -- Print some learning update details
       print('Learning update at episode: ' .. nIter)
       print('Step size: ' .. stepsize)
       print('Number of episodes in learning batch: ' .. numEps)
       print('Number of steps in learning batch: ' .. numSteps)
	end
	return learn
end

return getLearningUpdate
