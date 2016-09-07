local gum = require '../../../util/gym_utilities'()
local function learn(trajs, nIter, envDetails, tj, agent)
	theta, gradTheta = agent.model.actor:getParameters()
   -- Learn on the large set of timesteps in memory
   -- Concatenate the observations
   local numSteps = #tj.states
   local allObservations = torch.DoubleTensor(numSteps, envDetails.nbStates):zero()
   local allNextObservations = torch.DoubleTensor(numSteps, envDetails.nbStates):zero()
   local allActions = torch.DoubleTensor(numSteps, 1):zero()
   for i = 1, numSteps do
      allObservations[i] = tj.states[i]
      allActions[i] = tj.actions[i]
      allNextObservations[i] = tj.nextStates[i]
   end
   -- For each set of trajectories calculate compute the discounted sum of rewards
   local trajReturns = {}
   local trajNotTerminals = {}
   local trajLengths = torch.Tensor(#trajs):zero()
   for i = 1, #trajs do
      trajLengths[i], trajReturns[i], trajNotTerminals[i] = gum.discount(trajs[i],agent.gamma)
   end
   -- Compute the baseline
   local baseline = gum.getBaseline(trajs, trajLengths, trajReturns, agent.baselineType)
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
   
   -- TODO: there may be a better way to pass the actions along
   targetQ = agent.model.critic:forward(torch.cat(allNextObservations,agent.model.actor:forward(allNextObservations)))

   print(targetQ)
   
   -- Do the policy gradient update step
	local N = allObservations:size(1)
	local output = agent.model.actor:forward(allObservations)
	local advantagesNormalized = gum.whiten(allAdvantages)
	stepsize = agent.stepsizeStart * ((agent.nIterations - nIter) / agent.nIterations)
	
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
         targets[i] = ((output[i] - allActions[i])/(agent.policyStd^2)) * advantagesNormalized[i]
      end
   end
   -- Add to gradEntropy to targets to improve exploration and prevent convergence to potentially suboptimal deterministic policy
   targets:add(gradEntropy * agent.beta)
   agent.model.actor:backward(allObservations, targets)
   agent.gradThetaSq = agent.gradThetaSq * agent.weightDecay + torch.pow(gradTheta, 2) * (1 - agent.weightDecay)
   if agent.gradClip > 0 then
      gradTheta:clamp(-agent.gradClip, agent.gradClip)
   end
   -- tune the stepsize down as learning continues
   -- print('Step size ' .. stepsize)
   theta:add(torch.cdiv(gradTheta * stepsize, torch.sqrt(agent.gradThetaSq) + 1e-20))
end
return learn