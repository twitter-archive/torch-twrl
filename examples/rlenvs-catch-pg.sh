#! /bin/bash

clear
echo "REINFORCE on Catch Environment"
echo "************************************"

th run.lua \
   -env 'Catch' \
   -base 'rlenvs' \
   -policy categorical \
   -learningUpdate reinforce \
   -model mlp \
   -optimAlpha 0.9 \
   -timestepsPerBatch 1000 \
   -stepsizeStart 0.001 \
   -gamma 1 \
   -epsilon 0.01 \
   -epsilonDecayRate 0.90 \
   -nHiddenLayerSize 50 \
   -gradClip 5 \
   -baselineType padTimeDepAvReturn \
   -beta 0.01 \
   -weightDecay 0 \
   -windowSize 10 \
   -nSteps 1000 \
   -nIterations 10000 \
   -video 100 \
   -optimType rmsprop \
   -verboseUpdate true \
   -uploadResults false \
   -renderAllSteps true \
   -zoom 10
