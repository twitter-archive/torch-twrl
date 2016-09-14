#! /bin/bash

clear
echo "TD(Lmabda) SARSA agent on CartPole Environment"
echo "************************************"

th run.lua \
	-env 'CartPole-v0' \
	-policy egreedy \
	-learningUpdate tdLambda \
   -model qFunction \