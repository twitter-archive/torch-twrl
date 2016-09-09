#! /bin/bash

clear
echo "Testing Gym environments and agents:"
echo "*************************************************"

# th test_gym.lua --env CartPole-v1 --policy egreedy --epsilon 1 --epsilonDecayRate 0.9999999 --model qFunction --learningType noBatch --nIterations 1000 --nSteps 300 --renderAllSteps false --video 100 --learningUpdate tdLambda --numTiles 8 --numTilings 2 --lambda 0.9 --gamma 0.98 --force true --alphaScaleFactor 0.5 --initialWeightVal 0 --tdLearnUpdate qLearning
# th test_gym.lua --env MountainCar-v0 --policy egreedy --epsilon 1 --epsilonDecayRate 0.99999 --model qFunction --learningType noBatch --nIterations 1000 --nSteps 500 --renderAllSteps false --video 500 --learningUpdate tdLambda --numTiles 16 --numTilings 4 --lambda 0.9 --gamma 0.98 --force true --alphaScaleFactor 1 --initialWeightVal 0 --tdLearnUpdate qLearning
# echo "*************************************************"
# echo "*************************************************"
# echo "*************************************************"
# # 50000 max number of episodes.

th test_gym.lua -env 'Pendulum-v0' -nSteps 30 -nIterations 5 -timestepsPerBatch 30 -policy ddpgActor -model ddpgActorCritic -learningUpdate ddpgUpdate --uploadResults false -video 0 -renderAllSteps true -showTrajectory false

# echo "Gym Environment: MountainCar-v0"

# th test_gym.lua -env 'Pendulum-v0' -nSteps 10000 -nIterations 200 -timestepsPerBatch 1000 -policy normal -model singleHiddenLayerNormal -learningUpdate reinforce --uploadResults false -video 0 -showTrajectory true
# th test_gym.lua -env 'Pendulum-v0' -policy normal -learningUpdate reinforce -nSteps 1000 -nIterations 2 -model singleHiddenLayerNormal -timestepsPerBatch 1000 -stepsizeStart 0.01 -gamma 1 -nHiddenLayerSize 10 -video 2000 -gradClip 5 -baselineType padTimeDepAvReturn

# th test_gym.lua -env 'InvertedPendulum-v2' -policy normal -learningUpdate reinforce -nSteps 500 -nIterations 200 -model singleHiddenLayerNormal -timestepsPerBatch 1000 -stepsizeStart 0.3 -gamma 1 -nHiddenLayerSize 32 -video 5 -gradClip 5 -baselineType padTimeDepAvReturn --uploadResults false --renderAllSteps false --showTrajectory false
# th test_gym.lua -env 'CartPole-v0' -policy categorical -learningUpdate reinforce -nSteps 1000 -nIterations 200 -model singleHiddenLayerCategorical -timestepsPerBatch 1000 -stepsizeStart 0.3 -gamma 1 -nHiddenLayerSize 10 -video 5 -gradClip 5 -baselineType padTimeDepAvReturn
# echo "*************************************************"
# echo "Testing Gym: classic control environments with a random agent:"
# th test_gym.lua -env 'CartPole-v0' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'CartPole-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Acrobot-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'MountainCar-v0' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'MountainCarContinuous-v0' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Pendulum-v0' -policy random -model noModel -learningUpdate noLearning --uploadResults false

# echo "*************************************************"
# echo "Testing Gym: MuJoCo environments with a random agent:"
# th test_gym.lua -env 'InvertedPendulum-v1' -nSteps 1000 -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'InvertedDoublePendulum-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Reacher-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Swimmer-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Hopper-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Walker2d-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Ant-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Humanoid-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'HumanoidStandup-v1' -policy random -model noModel -learningUpdate noLearning --uploadResults false

# echo "*************************************************"
# echo "Testing Gym: Box2D environments with a random agent:"
# th test_gym.lua -env 'LunarLander-v2' -nSteps 1000 -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'CarRacing-v0' -nSteps 1000 -video 0 -policy random -model noModel -learningUpdate noLearning --uploadResults false

# echo "*************************************************"
# echo "Testing Gym: Toy text environments with a random agent:"
# th test_gym.lua -env 'FrozenLake-v0' -nSteps 1000 -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'FrozenLake8x8-v0' -nSteps 1000 -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Taxi-v1' -nSteps 1000 -policy random -model noModel -learningUpdate noLearning --uploadResults false
# th test_gym.lua -env 'Roulette-v0' -nSteps 1000 -policy random -model noModel -learningUpdate noLearning --uploadResults false

# echo "*************************************************"
# echo "Testing Gym: Atari environments with a random agent:"
# th test_gym.lua -env 'AirRaid-v0' -nSteps 1000 -video 5 -showTrajectory true -policy random -model noModel -learningUpdate noLearning --uploadResults false