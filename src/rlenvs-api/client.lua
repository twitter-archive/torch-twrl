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
    self.nSteps = opts.nSteps
    self.stepCounter = 1
    local Env = require('rlenvs.' .. env_id)
    self.env = Env(opts)
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
    return observation
end

function rlEnvsClient:env_step(instance_id, action, render, video_callable)
    local reward, obs, done = self.env:step(action)
    local endOfSteps = self.stepCounter % self.nSteps == 0
    if self.nSteps and endOfSteps then
        done = true
    elseif not endOfSteps then
        done = false
    end
    self.stepCounter = self.stepCounter + 1
    return obs, reward, done
end

function rlEnvsClient:env_action_space_info(instance_id)
    local actionSpec = self.env:getActionSpec()
    local action = {}
    action['name'] = actionSpec[1] == 'int' and 'Discrete' or 'Box'
    action['n'] = actionSpec[3][2] + 1 -- TODO we just select the MAX value
    return action
end

function rlEnvsClient:env_action_space_sample(instance_id)
    self.actionSpec = self.actionSpec or self:env_action_space_info()
    local action = torch.random(0, self.actionSpec['n'] - 1) -- TODO this seems hacky, make sure this works with all envs
    return action
end

function rlEnvsClient:env_observation_space_info(instance_id)
    local stateSpec = self.env:getStateSpec()
    local state = {}
    if torch.type(stateSpec[1]) ~= 'table' then
        -- it is one entry not multiple
        state['name'] = stateSpec[1] == 'int' and 'Discrete' or 'Box'
        state['shape'] = { stateSpec[2] }
        state['low'] = { stateSpec[3][1] }
        state['high'] = { stateSpec[3][2] }
    else
        -- multiple entries
        local low = {}
        local high = {}
        for index, value in ipairs(stateSpec) do
            state['name'] = value[1] == 'int' and 'Discrete' or 'Box' -- TODO hoping here that they all have the same discrete or box name
            low[#low + 1] = value[3][1] or math.huge
            high[#high + 1] = value[3][2] or math.huge
            state['shape'] = { index } -- TODO the size will be given by this index, we assume each state is a value
        end
        state['low'] = low
        state['high'] = high
    end
    return state
end

return m