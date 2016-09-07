-- randomDiscrete.lua
local function getAgent(opt)
	local opt = opt or {}
	local env = opt.env
	local agent = {}
	function agent.observe(state)
		return nil
	end
	function agent.selectAction(obsv)
		local actionSpec = env.getActionSpec()
	 	return math.floor(torch.random(actionSpec[3][1], actionSpec[3][2]))
	end
	function agent.learn()
		return nil
	end
	return agent
end
return getAgent