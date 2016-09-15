#! /bin/bash

clear
echo "REINFORCE on CartPole Environment"
echo "************************************"

th run.lua \
	-env 'CartPole-v0' \
	-policy categorical \
	-learningUpdate reinforce \
   -model mlp \
	-optimAlpha 0.9 \
   -timestepsPerBatch 200 \
	-stepsizeStart 0.2 -gamma 1 \
	-nHiddenLayerSize 10 \
	-gradClip 5 \
	-baselineType padTimeDepAvReturn \
	-beta 0.01 \
	-weightDecay 0 \
	-windowSize 10 \
   -nSteps 1000 \
	-nIterations 100 \
	-video 0 \
	-uploadResults true \
	-renderAllSteps false