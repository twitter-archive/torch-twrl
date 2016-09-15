local smallEps = 1e-8

-- learning update utilities
local function discount(traj, gamma)
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

local function getBaseline(trajs, trajLengths, trajReturns, baselineType)
   local maxLength = trajLengths:max()
   local paddedReturns = torch.DoubleTensor(#trajs,maxLength):zero()
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

function whiten(advantages)
   return (advantages - advantages:mean())/(advantages:std() + smallEps)
end

return {
   smallEps = smallEps,
   discount = discount,
   getBaseline = getBaseline,
   whiten = whiten
}