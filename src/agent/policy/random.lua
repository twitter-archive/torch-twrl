local function getPolicy(opt)
	local function selectAction(state, actionSampler)
		print(actionSampler)
      return actionSampler()
   end
   return selectAction
end
return getPolicy