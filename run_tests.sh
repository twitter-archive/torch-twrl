#! /bin/bash

clear
echo "Testing REINFORCE on CartPole Environment"
echo "*************************************************"

# th testScript.lua -env 'MountainCar-v0' -policy random -learningUpdate noLearning -model noModel -nSteps 1000 -nIterations 500 -timestepsPerBatch 1000 -video 200 -renderAllSteps true

th testScript.lua -env 'CartPole-v0' -policy categorical -learningUpdate \
	reinforce -model singleHiddenLayerCategorical -nSteps 1000 \
	-nIterations 500 -timestepsPerBatch 2000 -stepsizeStart 0.5 -gamma 1 \
	-nHiddenLayerSize 20 -video 0 -gradClip 5 -baselineType padTimeDepAvReturn \
	-renderAllSteps false -beta 0.01 -weightDecay 0.9 -windowSize 20
	-uploadResults true