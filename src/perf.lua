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
    --episodeRewards:zero()
  end

  local function addReward(nIter, reward, terminal)
    trajs[nIter] = trajs[nIter] or {}
    table.insert(traj, reward)
    if terminal then
      table.insert(trajs[nIter], traj)
      traj = {}
    end
  end

  local function getSummary()
    -- local episodeRewards = torch.Tensor(windowSize):zero()
    -- local lastNEpisodeRewards = torch.Tensor(windowSize):zero()
    -- local lastNEpisodeLengths = torch.Tensor(windowSize):zero()
    
    -- -- get last 10 episode rewards and lengths
    -- local count = 1
    -- for i=#trajs,1,-1 do
    --   for j = #trajs[i],1, -1 do
    --     if count > 10 then break end
    --     local episode = trajs[i][j]
    --     local length = #episode
    --     local rewardSum = mo.reduce(episode, function(a, v) return a + v end)
    --     lastNEpisodeLengths[count] = length
    --     lastNEpisodeRewards[count] = rewardSum
    --     count = count + 1
    --   end
    -- end
   
      local episodeRewards = torch.Tensor(#trajs):zero()
      local episodeLengths = torch.Tensor(#trajs):zero()
      for i = 1,#trajs do
         episodeLengths[i] = #trajs[i]
         for j = 1,#trajs[i] do
            episodeRewards[i] = episodeRewards[i] + trajs[i][j][1]
         end
      end

    local summary = {
       -- maxLength = lastNEpisodeLengths:max(),
       -- meanLength = lastNEpisodeLengths:mean(),
       -- maxReward = lastNEpisodeRewards:max(),
       -- meanReward = lastNEpisodeRewards:mean()
       numTrajectories = #trajs,
       meanLength = episodeLengths:mean(),
       meanReward = episodeRewards:mean(),
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
