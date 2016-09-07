local function cpEnv(opt)
   local opt = opt or {}
   local state
   local env = {}
   local gravity = opt.gravity or 9.8
   local cartMass = opt.cartMass or 1.0
   local poleMass = opt.poleMass or 0.1
   local poleLength = opt.poleLength or 0.5 -- half of pole length
   local forceMagnitude = opt.forceMagnitude or 10.0
   local tau = opt.tau or 0.02 -- seconds between state updates
   local totalMass = cartMass + poleMass
   local poleMassLength = poleMass * poleLength
   local x = 0 -- Cart position (m)
   local xDot = 0 -- Cart velocity
   local theta = 0 -- Pole angle (rad)
   local thetaDot = 0 -- Pole angular velocity
   -- 4 states returned, of type 'real', of dimensionality 1, with differing ranges
   function env.getStateSpec()
      return {
         {'real', 1, {-2.4, 2.4}},                    -- Cart position
         {'real', 1, {nil, nil}},                     -- Cart velocity
         {'real', 1, {math.rad(-12), math.rad(12)}},  -- Pole angle
         {'real', 1, {nil, nil}}                      -- Pole angular velocity
      }
   end
   -- 1 action required, of type 'int', of dimensionality 1, between 0 and 1 (left, right)
   function env.getActionSpec()
      return {'int', 1, {0, 1}}
   end
   -- Resets the cart
   function env.start()
      -- Reset position, angle and velocities
      x = 0 -- Cart position (m)
      xDot = 0 -- Cart velocity
      theta = 0 -- Pole angle (rad)
      thetaDot = 0 -- Pole angular velocity
      return {x, xDot, theta, thetaDot}
   end
   -- Drives the cart
   function env.step(action)
      -- Calculate acceleration
      local force = action == 1 and forceMagnitude or -forceMagnitude
      local cosTheta = math.cos(theta)
      local sinTheta = math.sin(theta)
      local temp = (force + poleMassLength * math.pow(thetaDot, 2) * sinTheta) / totalMass
      local thetaDotDot = (gravity * sinTheta - cosTheta * temp) / (poleLength * (4/3 - poleMass * math.pow(cosTheta, 2) / totalMass))
      local xDotDot = temp - poleMassLength * thetaDotDot * cosTheta / totalMass
      -- Update state using Euler's method
      x = x + tau * xDot
      xDot = xDot + tau * xDotDot
      theta = theta + tau * thetaDot
      thetaDot = thetaDot + tau * thetaDotDot
      -- Check failure (if cart reaches sides of track/pole tips too much)
      local reward = 1
      local terminal = false
      if (x < -2.4 or x > 2.4) or (theta < math.rad(-12) or theta > math.rad(12)) then
         reward = 0
         terminal = true
      end
      return reward, {x, xDot, theta, thetaDot}, terminal
   end
   return env
end
return cpEnv