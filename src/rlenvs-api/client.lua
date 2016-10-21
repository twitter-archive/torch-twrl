local rlEnvsClient = {}

local m = {}
function m.new(remote_base)
    local self = {}
    self.rlenvs = require 'rlenvs'
    self.remote_base = remote_base
    setmetatable(self, { __index = rlEnvsClient })
    return self
end

function rlEnvsClient:env_create(env_id, opts)
    self.env_id = env_id
    local Env = require('rlenvs.' .. env_id)
    self.env = Env(opts)
    if opts.renderAllSteps ~= 'false' then require 'image' self.qt = pcall(require, 'qt') end
    return self.env
end

function rlEnvsClient:env_list_all()
    local keyset = {}
    for k, v in pairs(self.rlenvs) do
        keyset[#keyset + 1] = tostring(v):match("<(.-)>") -- extract class name
    end
    return keyset
end

function rlEnvsClient:env_monitor_start(instance_id, directory, force, resume, video_callable)
end

function rlEnvsClient:env_monitor_close(instanceID)
end

function rlEnvsClient:env_reset(instance_id, opts)
    local observation = self.env:start(opts)
    self.window = self.qt and image.display({ image = observation, zoom = 10 })
    return observation
end

function rlEnvsClient:env_step(instance_id, action, render, video_callable)
    local reward, obs, done = self.env:step(action)
    if self.qt then
        image.display({ image = obs, zoom = 10, win = self.window })
    end
    return obs, reward, done
end

function rlEnvsClient:env_action_space_info(instance_id)
    return self.env:getActionSpace()
end

function rlEnvsClient:env_action_space_sample(instance_id)
    self.actionSpace = self.actionSpace or self:env_action_space_info()
    local action = torch.random(0, self.actionSpace['n'] - 1)
    return action
end

function rlEnvsClient:env_observation_space_info(instance_id)
    return self.env:getStateSpace()
end

return m