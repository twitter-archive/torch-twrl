local function getPolicy(opt)
	local randomActionSampler = opt.randomActionSampler
	local function selectAction(state)
      return randomActionSampler()
   end
   return selectAction
end
return getPolicy