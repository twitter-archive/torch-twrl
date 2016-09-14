local testSuite = torch.TestSuite()
local tester = torch.Tester()
local gymClient = require '../src/gym-http-api/binding-lua/gym_http_client'

local verbose = false
local render = false
local video_callable = 2

local runTest = require '../src/gym-http-api/binding-lua/test_api'({
   gymClient = gymClient, 
   verbose = verbose,
   render = render,
   video_callable = video_callable
})

-- Load all 
local util = require 'rl.util'()

function testSuite.torchTensorTest()
   local a = {2, torch.Tensor{1, 2, 2}}
   local b = {2, torch.Tensor{1, 2, 2.001}}
   tester:eq(a, b, 0.01, "a and b should be approximately equal")
end

function testSuite.testCartPole()
	local success = runTest('CartPole-v0')
   tester:eq(success, true, "testCartPole shouldn't give an error")
end

function testSuite.testPendulum()
   local success = runTest('Pendulum-v0')
   tester:eq(success, true, "testPendulum shouldn't give an error")
end

function testSuite.testFrozenLake()
   local success = runTest('FrozenLake-v0')
   tester:eq(success, true, "testCartPole shouldn't give an error")
end

function testSuite.testAtari()
   local success = runTest('BattleZone-v0')
   tester:eq(success, true, "testAtari shouldn't give an error if you have Atari configured")
end

function testSuite.testMujoco()
   local success = runTest('InvertedPendulum-v1')
   tester:eq(success, true, "testMujoco shouldn't give an error if you have MuJoCo configured")
end

function testSuite.tilecode()
	local numTilings = 8
   local numTiles = 32
	local memorySize = numTiles * numTiles
	local stateScalingFactor = {1, 1}
	local tc = require 'rl.agent.model.tilecoding'(({
		numTilings = numTilings, 
		memorySize = memorySize, 
		scaleFactor = stateScalingFactor
	}))
	local state = {3.6, 7.21}
   local tiles = tc.tiles(memorySize, numTilings, state)
   local fTiles = tc.feature(state)
   tester:eq(tiles, fTiles, "tiles and featuredTiles should be equal")
end

function testSuite.tilecodePredictable()
   local numTilings = 8
   local numTiles = 32
   local memorySize = numTiles * numTiles
   local stateScalingFactor = {1, 1}
   local tc = require 'rl.agent.model.tilecoding'(({
      numTilings = numTilings, 
      memorySize = memorySize, 
      scaleFactor = stateScalingFactor
   }))
   local state = {3.6, 7.21}
   local tiles = tc.tiles(memorySize, numTilings, state)
   local predictTables = {820, 119, 115, 465, 458, 260, 512, 505}
   tester:eq(tiles, predictTables, "tiles and predictTables should be equal")
end

tester:add(testSuite)
tester:run()