local function perf(opt)
   local mo = require 'moses'
   local opt = opt or {}
   local nIterations = opt.nIterations
   local iteration = 1
   local trajs = {}
   local windowSize = opt.windowSize or 10
   
   local episodeTrajectory = {}
   local episodeRewards = {}
   local episodeLengths = {}

   local function reset()
      episodeTrajectory = {}
      return episodeTrajectory
   end

   local function addReward(nIter, reward, terminal)
      trajs[nIter] = trajs[nIter] or {}
      table.insert(episodeTrajectory, reward)
      if terminal then
         -- add the episode trajectory to the table of all episode trajectories
         table.insert(trajs[nIter], episodeTrajectory)
         
         -- save episode length
         local epLength = #episodeTrajectory
         table.insert(episodeLengths, epLength)
         
         -- save episode total reward
         local epReward = 0
         for j = 1, epLength do
            epReward = epReward + episodeTrajectory[j]
         end
         table.insert(episodeRewards, epReward)

         -- reset episode trajectory table
         traj = reset()
      end
      return traj, trajs
   end

   local function getSummary(nIter)
      local meanEpReward = torch.Tensor(episodeRewards):mean()
      local stdEpReward = torch.Tensor(episodeRewards):std()
      local maxEpReward = torch.Tensor(episodeRewards):max()
      local meanEpRewardWindow
      if nIter % windowSize == 0 then
         meanEpRewardWindow = (torch.Tensor(episodeRewards)[{{nIter-windowSize+1,nIter}}]):mean()
         stdEpRewardWindow = (torch.Tensor(episodeRewards)[{{nIter-windowSize+1,nIter}}]):std()
      end
      local summary = {
         iteration = #trajs,
         numEps = numEps,
         windowSize = windowSize,
         meanEpReward = meanEpReward,
         stdEpReward = stdEpReward,
         maxEpReward = maxEpReward,
         meanEpRewardWindow = meanEpRewardWindow,
         stdEpRewardWindow = stdEpRewardWindow
      }
      if nIter % windowSize == 0 then
         print(summary)
      end
      return summary
   end

   return {
      reset = reset,
      addReward = addReward,
      getSummary = getSummary
   }
end

return perf