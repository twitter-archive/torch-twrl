local function mcEnv(opt)
	local opt = opt or {}
	local state
	local env = {}
	-- 2 states returned, of type 'real', of dimensionality 1, with differing ranges
	function env.getStateSpec()
	  return {
	    {'real', 1, {-0.07, 0.07}}, -- Velocity
	    {'real', 1, {-1.2, 0.6}} -- Position
	  }
	end
	-- 1 action required, of type 'int', of dimensionality 1, between -1 and 1 (left, neutral, right)
	function env.getActionSpec()
	  return {'int', 1, {-1, 1}}
	end
	function env.start()
		-- Reset the state
		state = {0, -0.5}
		return state
	end
	function env.step(action)
		local terminal = false
		-- Get the current state
		local velocity, position = state[1], state[2]
		local height = math.sin(3*position)
		-- Update velocity and position
		velocity = velocity + 0.001*action - 0.0025*math.cos(3*position)
		velocity = math.max(velocity, -0.07)
		velocity = math.min(velocity, 0.07)
		position = position + velocity
		position = math.max(position, -1.2)
		position = math.min(position, 0.6)
		-- Reset velocity if at very left (this is a hard wall)
		if position == -1.2 and velocity < 0 then
			velocity = 0
		end
		-- Update the state
		state = {velocity, position}
		-- Calculate reward
		local reward =  height-1
		-- Calculate termination
		local terminal = position >= 0.5 -- Car has made it over the (right) hill
		return reward, state, terminal
	end
	return env
end
return mcEnv