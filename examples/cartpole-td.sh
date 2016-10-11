#! /bin/bash

clear
echo "TD(Lambda) SARSA agent on CartPole Environment"
echo "************************************"

th /Users/korymathewson/Dropbox/work/torch-twrl/examples/run.lua \
   -env 'CartPole-v0' \
   -policy egreedy \
   -learningUpdate tdLambda \
   -model qFunction \
   -learningType noBatch \
   -epsilon 0.05 \
   -epsilonDecayRate 1 \
   -initialWeightVal -1 \
   -tdLearnUpdate qLearning \
   -relativeAlpha 0.05