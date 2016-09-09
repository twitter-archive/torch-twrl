local function getLearningUpdate(opt)
	local gum = require '../../../util/gym_utilities'()
	local opt = opt or {}
	local modelP = opt.model
	local model = modelP.model
	local envDetails = opt.envDetails
	local gamma = opt.gamma
	local baselineType = opt.baselineType
	local stepsizeStart = opt.stepsizeStart
	local policyStd = opt.policyStd
	local nIterations = opt.nIterations

	local function learn(trajs, tj)
		theta, gradTheta = model:getParameters()
   	-- Learn on the large set of timesteps in memory
   	-- Concatenate the observations

   	local numSteps = #tj.states
   	local allObservations = torch.DoubleTensor(numSteps, envDetails.nbStates):zero()
   	local allActions = torch.DoubleTensor(numSteps, 1):zero()

   	for i = 1, numSteps do
      allObservations[i] = tj.states[i]
      allActions[i] = tj.actions[i]
   	end
   	-- For each set of trajectories calculate compute the discounted sum of rewards
   	local trajReturns = {}
   	local trajNotTerminals = {}
   	local trajLengths = torch.Tensor(#trajs):zero()
   	for i = 1, #trajs do
      trajLengths[i], trajReturns[i], trajNotTerminals[i] = gum.discount(trajs[i],gamma)
   	end

		-- Compute the baseline
   	local baseline = gum.getBaseline(trajs, trajLengths, trajReturns, baselineType)
   	-- Compute the advantage function
   	-- Calculate variance-reduced reward (advantage)
   	local advs = {}
   	for i = 1, #trajs do
      advs[i] = trajReturns[i] - baseline[{{1,(#trajReturns[i])[1]}}]
   	end
   	local allAdvantages = torch.DoubleTensor(advs[1])
   	for i = 1, #trajs-1 do
      allAdvantages = torch.cat(allAdvantages, advs[i+1])
   	end

   	-- Do the policy gradient update step
		local N = allObservations:size(1)
		local output = model:forward(allObservations)
		local advantagesNormalized = gum.whiten(allAdvantages)
		stepsize = stepsizeStart --* ((nIterations - nIter) / nIterations)

   	-- Calculate (negative of) gradient of entropy of policy (for gradient descent): -(-logp(s) - 1)
		local gradEntropy = torch.log(output) - 1

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
      targets = torch.DoubleTensor(N,envDetails.nbActionSpace):zero()
      for i = 1, N do
         targets[i] = ((output[i] - allActions[i])/(policyStd^2)) * advantagesNormalized[i]
      end
   	end
   	-- Add to gradEntropy to targets to improve exploration and prevent convergence to potentially suboptimal deterministic policy
   	targets:add(gradEntropy * opt.beta)
   	model:backward(allObservations, targets)
   	modelP.gradThetaSq = modelP.gradThetaSq * opt.weightDecay + torch.pow(gradTheta, 2) * (1 - opt.weightDecay)
   	if opt.gradClip > 0 then
      modelP.gradTheta:clamp(-opt.gradClip, opt.gradClip)
   	end
   	-- tune the stepsize down as learning continues
   	-- print('Step size ' .. stepsize)
   	modelP.theta:add(torch.cdiv(modelP.gradTheta * stepsize, torch.sqrt(modelP.gradThetaSq) + 1e-20))
	end
	return learn
end

return getLearningUpdate
