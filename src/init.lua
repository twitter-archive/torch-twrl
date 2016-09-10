local rl = {}

-- Meta info
rl.VERSION = '0.1'
rl.LICENSE = 'MIT'

-- Utility packages 
rl.util = require 'rl.util'
rl.perf = require 'rl.perf'
rl.trajectory = require 'rl.trajectory'
rl.tilecoding = require 'rl.tilecoding'

rl.agent = require 'rl.agent'

-- Return package
return rl