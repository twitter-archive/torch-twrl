#! /bin/bash

clear
echo "Testing REINFORCE on CartPole Environment"
echo "*************************************************"
th testScript.lua -env 'CartPole-v0' -policy categorical -learningUpdate reinforce -model singleHiddenLayerCategorical -nSteps 1000 -nIterations 200 -timestepsPerBatch 1000 -stepsizeStart 0.3 -gamma 1 -nHiddenLayerSize 10 -video 200 -gradClip 5 -baselineType padTimeDepAvReturn