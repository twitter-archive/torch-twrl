local rl = {}

-- Meta info
rl.VERSION = '0.1'
rl.LICENSE = 'MIT'

-- Utility packages 
rl.agent = require 'rl.agent'
rl.util = require 'rl.util'
rl.perf = require 'rl.perf'
rl.experiment = require 'rl.experiment'
rl.gymClient = require 'rl.binding-lua.gym_http_client'

-- Return package
return rl