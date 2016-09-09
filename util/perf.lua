local function perf(opt)

  local opt = opt or {}
  local nIterations = opt.nIterations
  local episodeRewards = torch.Tensor(nIterations):zero()

  local function reset()
    episodeRewards:zero()
  end

  local function addReward(nIter, reward)
    episodeRewards[nIter] = episodeRewards[nIter] + reward
  end

  local function getSummary()
    local summary = {
      max = episodeRewards:max(),
      mean = episodeRewards:mean()
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
