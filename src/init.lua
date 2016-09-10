local rl = {}

-- Meta info
rl.VERSION = '0.1'
rl.LICENSE = 'MIT'

-- Utility packages 
rl.agent = require 'rl.agent'
rl.util = require 'rl.util'
rl.perf = require 'rl.perf'
rl.trajectory = require 'rl.trajectory'
rl.tilecoding = require 'rl.tilecoding'
rl.experiment = require 'rl.experiment'

-- Return package
return rl