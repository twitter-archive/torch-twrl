#! /bin/bash

clear
echo "Random agent on CartPole Environment"
echo "************************************"

th run.lua \
   -env 'CartPole-v0' \
   -policy random \
   -learningUpdate noLearning \
   -model noModel \
   -video 0 \
   -renderAllSteps false \
   -nIterations 10 \