#! /bin/bash

clear
echo "REINFORCE on Pendulum Environment"
echo "************************************"

th run.lua \
   -env 'Pendulum-v0' \
   -policy normal \
   -learningUpdate reinforce \
   -model mlp \
   -optimAlpha 0.9 \
   -timestepsPerBatch 1000 \
   -stepsizeStart 0.3 -gamma 1 \
   -nHiddenLayerSize 10 \
   -gradClip 5 \
   -baselineType padTimeDepAvReturn \
   -beta 0.01 \
   -weightDecay 0 \
   -windowSize 100 \
   -nSteps 1000 \
   -nIterations 500 \
   -video 0 \
   -uploadResults true \
   -renderAllSteps false