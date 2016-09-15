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
   -timestepsPerBatch 400 \
	-stepsizeStart 0.1 -gamma 0.9 \
	-nHiddenLayerSize 10 \
	-gradClip 10 \
	-baselineType padTimeDepAvReturn \
	-beta 0.1 \
	-weightDecay 0 \
	-windowSize 10 \
   -nSteps 1000 \
	-nIterations 50 \
	-video 0 \
	-uploadResults true \
	-renderAllSteps false