local function dgwEnv(opt)
   local opt = opt or {}
   local world = torch.Tensor(101, 101):fill(-0.5)
   local state
   local env = {}
   -- 2 states returned, ints defining x and y state
   function env.getStateSpec()
      return{
        {'int', 1, {1, 101}}, -- x-state
        {'int', 1, {1, 101}}, -- y-state
      }
   end
   -- 1 action required, of type 'int', of dimensionality 1, between 1 and 4
   function env.getActionSpec()
      return {'int', 1, {1, 4}}
   end
   -- Min and max reward
   function env.getRewardSpec(world)
      return torch.min(world), 0
   end
   -- Reset state
   function env.start()
      state = {20, 40}
      return state
   end
   -- Move up, right, down or left
   function env.step(action)
      local terminal = false
      -- Get the current state
      local s = {state[1],state[2]}
      -- Move
      if action == 1 then
      -- Move up
         s[2] = math.min(s[2] + 5, 101)
      elseif action == 2 then
         -- Move right
         s[1] = math.min(s[1] + 5, 101)
      elseif action == 3 then
         -- Move down
         s[2] = math.max(s[2] - 5, 1)
      else
         -- Move left
         s[1] = math.max(s[1] - 5, 1)
      end
      -- update the state
      state = {s[1],s[2]}
      -- Look up cost of moving to state
      local reward = world[{s[1],s[2]}]
      -- Check if reached goal
      if s[1] == 1 and s[2] == 1 then
         reward = 2
         terminal = true
      end
      return reward, s, terminal
   end
   return env
end
return dgwEnv