local testSuite = torch.TestSuite()
local tester = torch.Tester()
local gymClient = require '../src/gym-http-api/binding-lua/gym_http_client'

-- Load all 
local util = require 'rl.util'()

function testSuite.torchTensorTest()
   local a = {2, torch.Tensor{1, 2, 2}}
   local b = {2, torch.Tensor{1, 2, 2.001}}
   tester:eq(a, b, 0.01, "a and b should be approximately equal")
end

function testSuite.testAPI()
	local function testAPI()
		local verbose = false
		local runTest = require '../src/gym-http-api/binding-lua/test_api'({
			gymClient = gymClient, 
			verbose = verbose
		})
		local testEnvs = {'CartPole-v0', 'FrozenLake-v0'}
		for i = 1,#testEnvs do
			local success = runTest(testEnvs[i])
		end
	end
   tester:assertNoError(testAPI, "testAPI shouldn't give an error")
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