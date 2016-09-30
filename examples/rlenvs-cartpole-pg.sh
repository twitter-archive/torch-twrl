#! /bin/bash

clear
echo "REINFORCE on CartPole Environment"
echo "************************************"

th run.lua \
   -env 'CartPole' \
   -base 'rlenvs' \
   -policy categorical \
   -learningUpdate reinforce \
   -model mlp \
   -optimAlpha 0.9 \
   -timestepsPerBatch 1000 \
   -stepsizeStart 0.3 \
   -gamma 1 \
   -nHiddenLayerSize 10 \
   -gradClip 5 \
   -baselineType padTimeDepAvReturn \
   -beta 0.01 \
   -weightDecay 0 \
   -windowSize 10 \
   -nSteps 200 \
   -nIterations 1000 \
   -video 100 \
   -optimType rmsprop \
   -verboseUpdate true \
   -uploadResults false \
   -renderAllSteps false