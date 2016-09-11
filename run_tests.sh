#! /bin/bash

clear
echo "Testing REINFORCE on CartPole Environment"
echo "*************************************************"
th testScript.lua -env 'CartPole-v0' -policy categorical -learningUpdate reinforce -model singleHiddenLayerCategorical -nSteps 1000 -nIterations 200 -timestepsPerBatch 600 -stepsizeStart 0.4 -gamma 1 -nHiddenLayerSize 8 -video 50 -gradClip 6 -baselineType padTimeDepAvReturn -renderAllSteps false -beta 0.001 -weightDecay 0.8