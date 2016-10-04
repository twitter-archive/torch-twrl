#! /bin/bash
step_sizes=(0.2 0.15 0.1 0.075 0.05 0.025 0.01)
COUNTER=0
PORT="${PORT:-5000}"
while [ $COUNTER -lt 10 ]; do
echo Iteration number $COUNTER
for i in ${step_sizes[@]}; do
    echo ${i}
    ## may need to force pipe password into sudo, like this... echo 'password' | sudo -kS ls
    sudo lsof -i :$PORT | grep 'python\|Python' | cut -d " " -f3 | xargs kill -9
    # start the server
    python ../src/gym-http-api/gym_http_server.py -p $PORT & SERVER_PID=$!
    # give the server time to start
    sleep 2
    # run the experiment
    th run.lua \
       -env 'CartPole-v0' \
       -policy categorical \
       -learningUpdate reinforce \
       -model mlp \
       -optimAlpha 0.9 \
       -timestepsPerBatch 1000 \
       -stepsizeStart ${i} \
       -gamma 1 \
       -nHiddenLayerSize 10 \
       -gradClip 5 \
       -baselineType padTimeDepAvReturn \
       -beta 0.01 \
       -weightDecay 0 \
       -windowSize 10 \
       -nSteps 1000 \
       -nIterations 600 \
       -video 0 \
       -optimType rmsprop \
       -verboseUpdate true \
       -uploadResults false \
       -renderAllSteps false \
       -learningType batch \
       -gymHttpServer http://127.0.0.1:$PORT \
       -experimentLogName 20161003sweeplongNew
    # close the server cleanly
    kill $SERVER_PID
    sudo lsof -i :$PORT | grep 'python\|Python' | cut -d " " -f3 | xargs kill -9
done
let COUNTER=COUNTER+1
done