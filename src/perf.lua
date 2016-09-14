local function perf(opt)
  local mo = require 'moses'
  local opt = opt or {}
  local nIterations = opt.nIterations
  local iteration = 1
  local traj = {}
  local trajs = {}
  local windowSize = opt.windowSize or 10

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

  local function getSummary()
      local episodeRewards = torch.Tensor(#trajs):zero()
      local episodeLengths = torch.Tensor(#trajs):zero()
      for i = 1,#trajs do
         episodeLengths[i] = #trajs[i][1]
         for j = 1,#trajs[i][1] do
            episodeRewards[i] = episodeRewards[i] + trajs[i][1][j]
         end
      end

      local meanLength = 0
      local meanReward = 0
      local maxLength, maxReward

      if #trajs > windowSize then
         meanLength = episodeRewards[{{#trajs-windowSize,#trajs}}][1]
         meanReward = episodeRewards[{{#trajs-windowSize,#trajs}}][1]
      end

    local summary = {
       windowSize = windowSize,
       iteration = #trajs,
       meanLengthOverWindowSize = meanLength,
       meanRewardOverWindowSize = meanReward,
       maxLength = episodeLengths:max(),
       maxReward = episodeRewards:max(),
    }
    return summary
  end

  return {
    reset = reset,
    addReward = addReward,
    getSummary = getSummary
  }
end

return perf