#!/bin/sh
cd ./
make
./trainingSet.exe >& ./data/trainingSet1.txt
./neuralNet.exe >& outfile.txt
tail -1 outfile.txt > testFile.txt
tail -1 goodcases.txt > confirm.txt
diff testFile.txt confirm.txt
