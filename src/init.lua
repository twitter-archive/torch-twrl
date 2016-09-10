local rl = {}

-- Meta info
rl.VERSION = '0.1'
rl.LICENSE = 'MIT'

-- Utility packages 
rl.util = require 'rl.util'
rl.perf = require 'rl.perf'
rl.trajectory = require 'rl.trajectory'
rl.tilecoding = require 'rl.tilecoding'

-- -- Main packages
rl.agent = require 'rl.agent'

-- -- Agent packages
-- rl.agents.learningUpdate = require 'rl.agents.learningUpdate'
-- rl.agents.model = require 'rl.agents.model'
-- rl.agents.policy = require 'rl.agents.policy'

-- Return package
return rl