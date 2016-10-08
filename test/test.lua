local torch = require 'torch'
local base = torch.TestSuite()
local performance = torch.TestSuite()
local tilecoding = torch.TestSuite()

local tester = torch.Tester()

-- Load all 
local util = require 'twrl.util'()

function base.torchTensor()
   local a = {2, torch.Tensor{1, 2, 2}}
   local b = {2, torch.Tensor{1, 2, 2.001}}
   tester:eq(a, b, 0.01, "a and b should be approximately equal")
end

function tilecoding.tilecodeConsistent()
   local numTilings = 8
   local numTiles = 32
   local memorySize = numTiles * numTiles
   local tc = require 'twrl.agent.model.tilecoding'({})
   local state1 = {3.6, 7.21}
   local tiles1 = tc.tiles(memorySize, numTilings, state1)
   local state2 = {3.7, 7.21}
   local tiles2 = tc.tiles(memorySize, numTilings, state2)
   tester:ne(tiles1, tiles2, "tiles1 and tiles2 should not be equal")
end

function tilecoding.tilecodePredictable()
   local numTilings = 8
   local numTiles = 32
   local memorySize = numTiles * numTiles
   local tc = require 'twrl.agent.model.tilecoding'({})
   local state = {3.6, 7.21}
   local tiles1 = tc.tiles(memorySize, numTilings, state)
   local tiles2 = tc.tiles(memorySize, numTilings, state)
   tester:eq(tiles1, tiles2, "tiles1 and tiles 2 should be equal")
end

function performance.reset()
   local perf = require 'twrl.perf'()
   local emptyTable = perf.reset()
   tester:eq(emptyTable, {}, "performance: reset failed")
end

function performance.addRewardTerminal()
   local perf = require 'twrl.perf'()
   local traj, trajs = perf.addReward(1, 1, true)
   tester:eq(traj, {}, "performance: add reward terminal failed")
end

function performance.addRewardNonTerminal()
   local perf = require 'twrl.perf'()
   local traj, trajs = perf.addReward(1, 1, false)
   tester:eq(traj, {1}, "performance: add reward non-terminal failed")
end

function performance.getSummary()
   local perf = require 'twrl.perf'({windowSize = 10})
   local _, _ = perf.addReward(1, 0, false)
   local _, _ = perf.addReward(1, 0, false)
   local _, _ = perf.addReward(1, 1, true)
   local _, _ = perf.addReward(2, 0, false)
   local _, _ = perf.addReward(2, 0, false)
   local _, _ = perf.addReward(2, 1, true)
   local summary = perf.getSummary(2)
   tester:eq(summary.windowSize, 10, "performance: get summary failed")
end

tester:add(base)
tester:add(tilecoding)
tester:add(performance)
tester:run()