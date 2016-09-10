#! /bin/bash

clear
echo "Testing REINFORCE on CartPole Environment"
echo "*************************************************"
th test_gym.lua --env CartPole-v1 --policy egreedy --epsilon 1 --epsilonDecayRate 0.9999999 --model qFunction --learningType noBatch --nIterations 1000 --nSteps 300 --renderAllSteps false --video 100 --learningUpdate tdLambda --numTiles 8 --numTilings 2 --lambda 0.9 --gamma 0.98 --force true --alphaScaleFactor 0.5 --initialWeightVal 0 --tdLearnUpdate qLearning