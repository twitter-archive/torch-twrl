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

tester:add(testSuite)
tester:run()