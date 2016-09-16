local actionSamplers = require 'twrl.agent.policy.actionSamplers'

local policy = {
    egreedy = require 'twrl.agent.policy.egreedy',
    normal = require 'twrl.agent.policy.stochasticModelPolicy'({
             actionSampler = actionSamplers.normal
        }),
    categorical = require 'twrl.agent.policy.stochasticModelPolicy'({
             actionSampler = actionSamplers.categorical
        }),
    random = require 'twrl.agent.policy.random',
}

return policy