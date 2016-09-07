local function rwEnv(opt)
   local world
   local state
   local opt = opt or {}
   local env = {}
   -- 1 states returned, of type 'int', of dimensionality 1, between 1 and 6 (the terminal states)
   function env.getStateSpec()
      return {'int', 1, {1, 6}} -- Position
   end
   -- 1 action required, of type 'int', of dimensionality 1, between 1 and 2 (left or right)
   function env.getActionSpec()
      return {'int', 1, {1, 2}}
   end
   -- Min and max reward
   function env.getRewardSpec()
      return 0, 1
   end
   -- Reset state
   function env.start()
      state = 3
      return state
   end
   -- Move left or right
   function env.step(action)
      local reward = -0.1
      local terminal = false
      -- if action is 1 then move left
      if action == 1 then
         state = state - 1
         -- terminate if all the way left
         if state == 1 then
            terminal = true
         end
      else -- move right
         state = state + 1
         -- terminate if all the way right
         if state == env.getStateSpec()[3][2] then
            reward = 1
            terminal = true
         end
      end
      return reward, state, terminal
   end
   return env
end
return rwEnv