#!/bin/bash

if ! [ -d $PWD/solvers/Cashwmaxsat-CorePlus ]; then
  cd solvers
  wget https://maxsat-evaluations.github.io/2022/mse22-solver-src/complete/Cashwmaxsat-CorePlus.zip
  unzip Cashwmaxsat-CorePlus.zip
  cd Cashwmaxsat-CorePlus/bin
  chmod +x cashwmaxsatcoreplus
  cd ../../..
fi
export PATH=$PWD/solvers/Cashwmaxsat-CorePlus/bin:$PATH