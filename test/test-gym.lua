local torch = require 'torch'
local api = torch.TestSuite()
local atari = torch.TestSuite()
local mujoco = torch.TestSuite()
local experiment = torch.TestSuite()

local tester = torch.Tester()

local gymClient = require '../src/gym-http-api/binding-lua/gym_http_client'

-- disable rendering and video for testing
local verbose = false
local render = false
local video_callable = 0

local runTest = require '../src/gym-http-api/binding-lua/test_api'({
   gymClient = gymClient, 
   verbose = verbose,
   render = render,
   video_callable = video_callable
})

function api.testCartPole()
   local success = runTest('CartPole-v0')
   tester:eq(success, true, "testCartPole shouldn't give an error")
end

function api.testPendulum()
   local success = runTest('Pendulum-v0')
   tester:eq(success, true, "testPendulum shouldn't give an error")
end

function api.testFrozenLake()
   local success = runTest('FrozenLake-v0')
   tester:eq(success, true, "testCartPole shouldn't give an error")
end

function atari.testAtari()
   local success = runTest('BattleZone-v0')
   tester:eq(success, true, "testAtari shouldn't give an error if you have Atari configured")
end

function mujoco.testMujoco()
   local success = runTest('InvertedPendulum-v1')
   tester:eq(success, true, "testMujoco shouldn't give an error if you have MuJoCo configured")
end

function experiment.badExperimentCall()
   local performance = require 'rl.experiment'()
   tester:eq(performance, {}, "bad experiment call should fail with bad settings ")
end

function experiment.randomNoLearningNoModel()
   local env = 'CartPole-v0'
   local params = {}
   local agent = {
      policy = 'random',
      learningUpdate = 'noLearning',
      model = 'noModel'
   }
   local nSteps = 2
   local nIterations = 2
   local performance = require 'rl.experiment'(env, agent, nSteps, nIterations, params)
   tester:eq(performance.iteration, 2, "basic experiment should run")
end

tester:add(api)
tester:add(atari):disable('testAtari')
tester:add(mujoco):disable('testMujoco')
tester:add(experiment)
tester:run()