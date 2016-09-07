#! /bin/bash

clear
echo "Testing local environments:"

# -----------
echo "Environment: RandomWalk"
echo "Random agent"
th test_local.lua -env RandomWalk -agent randomDiscrete -nSteps 100 -nEpisodes 1000 -nReport 100
echo "SARSA agent"
th test_local.lua -env RandomWalk -agent sarsaDiscrete -nSteps 100 -nEpisodes 1000 -nReport 100 -alpha 0.2 -gamma 0.9 -epsilon 0.01
# -----------
echo "Environment: DiscreteGridWorld"
echo "Random agent"
th test_local.lua -env DiscreteGridWorld -agent randomDiscrete -nSteps 50 -nEpisodes 10000 -nReport 1000
echo "SARSA agent"
th test_local.lua -env DiscreteGridWorld -agent sarsaDiscrete -nSteps 50 -nEpisodes 10000 -nReport 1000 -alpha 0.2 -gamma 0.9 -epsilon 0.01
# -----------
echo "Environment: MountainCar"
echo "Random agent"
th test_local.lua -env MountainCar -agent randomDiscrete -nSteps 500 -nEpisodes 1000 -nReport 100
echo "SARSA-Lambda agent"
th test_local.lua -env MountainCar -agent sarsaLambda -nSteps 500 -nEpisodes 1000 -nReport 10 -gamma 1 -lambda 0.99 -epsilon 0 -epsilonDecayRate 1 -alphaScaleFactor 0.05 -numTiles 32 -numTilings 5
echo "Q-Lambda agent"
th test_local.lua -env MountainCar -agent qLambda -nSteps 500 -nEpisodes 1000 -nReport 100 -gamma 1 -lambda 0.9 -epsilon 0.1 -epsilonDecayRate 0.999 -alphaScaleFactor 0.05 -numTiles 32 -numTilings 4
# -----------
echo "Environment: CartPole"
echo "Random agent"
th test_local.lua -env CartPole -agent randomDiscrete -nSteps 100000 -nEpisodes 1000 -nReport 100
echo "SARSA-Lambda agent"
th test_local.lua -env CartPole -agent sarsaLambda -nSteps 300 -nEpisodes 1000 -nReport 100 -gamma 0.99 -lambda 0.9 -epsilon 1 -epsilonDecayRate 0.999 -alphaScaleFactor 0.5 -numTiles 32 -numTilings 4
echo "Q-Lambda agent"
th test_local.lua -env CartPole -agent qLambda -nSteps 300 -nEpisodes 1000 -nReport 100 -gamma 0.99 -lambda 0.9 -epsilon 1 -epsilonDecayRate 0.999 -alphaScaleFactor 0.5 -numTiles 32 -numTilings 4