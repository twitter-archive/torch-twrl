#! /bin/bash

clear
echo "Testing REINFORCE on CartPole Environment"
echo "*************************************************"

# th testScript.lua -env 'MountainCar-v0' -policy random -learningUpdate noLearning -model noModel -nSteps 1000 -nIterations 500 -timestepsPerBatch 1000 -video 200 -renderAllSteps true

th testScript.lua -env 'CartPole-v0' -policy categorical -learningUpdate reinforce -model singleHiddenLayerCategorical -nSteps 1000 -nIterations 500 -timestepsPerBatch 1000 -stepsizeStart 0.3 -gamma 1 -nHiddenLayerSize 10 -video 200 -gradClip 5 -baselineType padTimeDepAvReturn -renderAllSteps true -beta 0.01 -weightDecay 0.9