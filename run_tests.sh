#! /bin/bash

clear
echo "Testing REINFORCE on CartPole Environment"
echo "*************************************************"

# MuJoCo test
# th testScript.lua -env 'InvertedPendulum-v1' -policy random -learningUpdate noLearning -model noModel -nSteps 1000 -nIterations 500 -timestepsPerBatch 1000 -video 200 -renderAllSteps true

# Atari test
# th testScript.lua -env 'Assault-v0' -policy random -learningUpdate noLearning -model noModel -nSteps 1000 -nIterations 500 -timestepsPerBatch 1000 -video 200 -renderAllSteps true

th testScript.lua -env 'Pendulum-v0' -policy random -learningUpdate noLearning -model noModel -nSteps 1000 -nIterations 500 -timestepsPerBatch 1000 -video 200 -renderAllSteps true

# MuJoCo test
# th testScript.lua -env 'InvertedPendulum-v1' -policy random -learningUpdate noLearning -model noModel -nSteps 1000 -nIterations 500 -timestepsPerBatch 1000 -video 200 -renderAllSteps true

# # cartpole example https://gym.openai.com/evaluations/eval_48l1nOQ7ur6htkF9uGw
# th testScript.lua -env 'CartPole-v0' \
# 	-policy categorical -learningUpdate reinforce \
#    -model mlp -optimAlpha 0.9 \
#    -timestepsPerBatch 1000 -stepsizeStart 0.3 -gamma 1 \
# 	-nHiddenLayerSize 10 -gradClip 5 -baselineType padTimeDepAvReturn \
# 	-beta 0.01 -weightDecay 0 -windowSize 100 \
#    -nSteps 1000 -nIterations 1000 -video 0 \
# 	-uploadResults true -renderAllSteps false

# th testScript.lua -env 'CartPole-v0' \
#  -policy egreedy -learningUpdate tdLambda \
#  -model qFunction -epsilon 0.8 -epsilonDecayRate 0.9999 \
#  -epsilonMinValue 0.1 \
#  -tdLearnUpdate qLearning -windowSize 10 \
#  -timestepsPerBatch 1 -learningType noBatch \
#  -gamma 0.96 -lambda 0.9 \
#  -initialWeightVal -1 -traceType replacing \
#  -numTiles 40 -numTilings 4 -relativeAlpha 0.08 \
#  -nSteps 1000 -nIterations 2000 -video 20 \
#  -uploadResults true -renderAllSteps false