local function perf(opt)
  local mo = require 'moses'
  local opt = opt or {}
  local nIterations = opt.nIterations
  local iteration = 1
  local traj = {}
  local trajs = {}
  local windowSize = opt.windowSize
  local printPerfEvery = opt.printPerfEvery or windowSize

  local function reset()
    traj = {}
    return traj
  end

  local function addReward(nIter, reward, terminal)
    trajs[nIter] = trajs[nIter] or {}
    table.insert(traj, reward)
    if terminal then
      table.insert(trajs[nIter], traj)
      traj = {}
    end
    return traj, trajs
  end

  local function getSummary(nIter)
      local numEps = #trajs
      local episodeRewards = torch.Tensor(numEps):zero()
      local episodeLengths = torch.Tensor(numEps):zero()

      for i = 1,numEps do
         episodeLengths[i] = #trajs[i][1]
         for j = 1,#trajs[i][1] do
            episodeRewards[i] = episodeRewards[i] + trajs[i][1][j]
         end
      end

      local meanEpLength = episodeLengths:mean()
      local meanEpReward = episodeRewards:mean()
      local maxEpLength = episodeLengths:max()
      local maxEpReward = episodeRewards:max()

      -- if #trajs > windowSize then
      --    -- TODO: should handle the first iteration learning as well
      --    meanLength = episodeLengths[{{#trajs-windowSize,#trajs}}][1]
      --    meanReward = episodeRewards[{{#trajs-windowSize,#trajs}}][1]
      -- end

    local summary = {
      --TODO: do not need window size
       windowSize = windowSize,
       iteration = #trajs,
       episodeRewards = episodeRewards,
       episodeLengths = episodeLengths,
       meanEpLength = meanEpLength,
       meanEpReward = meanEpReward,
       maxEpLength = maxEpLength,
       maxEpReward = maxEpReward
    }
    if nIter % printPerfEvery == 0 then
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