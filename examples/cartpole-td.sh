#! /bin/bash

clear
echo "TD(Lambda) agent on CartPole Environment"
echo "************************************"

th run.lua \
   -env 'CartPole-v0' \
   -policy egreedy \
   -learningUpdate tdLambda \
   -model qFunction \
   -learningType noBatch \
   -epsilon 0.2 \
   -epsilonDecayRate 0.9999 \
   -initialWeightVal 0 \
   -tdLearnUpdate SARSA \
   -relativeAlpha 0.05
