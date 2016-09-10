local function utilities()
   local eps = 1e-20
   local util = {}
   function util.getStateAndActionSpecs(stateSpec, actionSpec)
      -- Get environment specifications in case they are needed
      local nbStates, nbStatesDiscrete
      if stateSpec['name'] == 'Discrete' then
         nbStates = 1
         nbStatesDiscrete = stateSpec['n']
      else
         totalStateSpace = 1
         for i = 1, #stateSpec['shape'] do
            totalStateSpace = totalStateSpace * stateSpec['shape'][i]
         end
         nbStates = totalStateSpace
      end
      local nbActions, nbActionSpace = 0
      local actionType
      local actionSpaceBounds
      if actionSpec['name'] == 'Discrete' then
         nbActions = actionSpec['n']
         actionType = 'Discrete'
      else
         nbActionSpace = actionSpec['shape'][1]
         actionType = 'Box'
         actionSpaceBounds = torch.Tensor(nbActionSpace, 3):zero()
         for i = 1, nbActionSpace do
            actionSpaceBounds[i][1] = actionSpec['low'][i]
            actionSpaceBounds[i][2] = actionSpec['high'][i]
            actionSpaceBounds[i][3] = actionSpaceBounds[i][2] - actionSpaceBounds[i][1]
         end
         nbActions = nbActionSpace
      end
      envDetails = {
         stateSpec = stateSpec,
         nbStates = nbStates,
         nbStatesDiscrete = nbStatesDiscrete,
         totalStateSpace = totalStateSpace,
         actionSpec = actionSpec,
         actionType = actionType,
         actionSpaceBounds = actionSpaceBounds,
         nbActionSpace = nbActionSpace,
         nbActions = nbActions
      }
      return envDetails
   end
   function util.discount(traj, gamma)
      -- Given a trajectory, computes a discounted return such that
      -- y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ..
      local trajReturn = torch.Tensor(#traj)
      local trajNotTerminal = torch.Tensor(#traj):zero()
      trajReturn[#traj] = traj[#traj].reward
      trajNotTerminal[#traj] = 1 - traj[#traj].terminal
      for j = #traj-1, 1, -1 do
         trajReturn[j] = traj[j].reward + gamma*trajReturn[j+1]
         trajNotTerminal[j] = 1 - traj[j].terminal
      end
      return #traj, trajReturn, trajNotTerminal
   end
   function util.getBaseline(trajs, trajLengths, trajReturns, baselineType)
      local maxLength = trajLengths:max()
      local paddedReturns = torch.DoubleTensor(#trajs,maxLength):zero()
      -- Compute the time dependent baseline
      if baselineType == 'padTimeDepAvReturn' then
         for i = 1, #trajs do
            if trajLengths[i] < maxLength then
               paddedReturns[i] = torch.cat(trajReturns[i],torch.DoubleTensor(maxLength - trajLengths[i]):zero())
            else
               paddedReturns[i] = trajReturns[i]
            end
         end
      end
      return paddedReturns:mean(1):t():squeeze()
   end
   function util.whiten(advantages)
      return (advantages - advantages:mean())/(advantages:std() + eps)
   end
   return util
end
return utilities