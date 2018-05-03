#!/bin/bash

mkdir data
wget http://nlg.csie.ntu.edu.tw/nlpresource/NTUSD-Fin/NTUSD-Fin.zip
unzip -u NTUSD-Fin.zip
mv NTUSD-Fin/*.json data/
rm -rf NTUSD-Fin.zip NTUSD-Fin
