local function trajectory()
   local tds = require 'tds'
   local tj = {}
   tj.states = tds.Vec()
   tj.actions = tds.Vec()
   tj.rewards = tds.Vec()
   tj.nextStates = tds.Vec()
   tj.terminals = tds.Vec()
   function tj.pushTraj(traj)
      for i = 1, #traj do
         tj.states[#tj.states + 1] = traj[i]['state']
         tj.actions[#tj.actions + 1] = traj[i]['action']
         tj.rewards[#tj.rewards + 1] = traj[i]['reward']
         tj.nextStates[#tj.nextStates + 1] = traj[i]['nextState']
         tj.terminals[#tj.terminals + 1] = traj[i]['terminal']
      end
   end
   function tj.getNumTrajs()
      return(#tj.states)
   end
   function tj.clearTrajs()
      tj.states = tds.Vec()
      tj.actions = tds.Vec()
      tj.rewards = tds.Vec()
      tj.nextStates = tds.Vec()
      tj.terminals = tds.Vec()
   end
   return tj
end
return trajectory