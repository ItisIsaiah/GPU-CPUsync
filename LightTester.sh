#!/bin/bash
size=100
n=5

taskset -c 0 ./Matrix -size $size -n $n -sync block >Tblock.txt 
wait
taskset -c 0 ./Matrix -size $size -n $n -sync spin >TSpin.txt 
wait


gnuplot -persist <<-EOFMarker
    set title "Time vs Size without sysbench" 
    
    set datafile separator "|"
    
    plot "TSpin.txt" with lines, "Tblock.txt" with lines
EOFMarker