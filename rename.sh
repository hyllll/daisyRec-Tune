#!/bin/bash

cd ./experiment_datasets
for file_a in `ls *.dat`;do
    string=${file_a}
     array=(${string//_/ })
     if [ "${array[2]}" = "5core" ]
     then
         tmp="5filter"
     elif [ "${array[2]}" = "10core" ]
     then
         tmp="10filter"
     else
         continue
     fi
    file_b=${array[0]}"_"${array[1]}"_"${tmp}"_"${array[3]}
    mv $file_a $file_b
#      echo ${file_b}
done