#!/bin/bash
size=1000
n=5



taskset -c 0 ./Matrix -size $size -n $n -sync spin > "$(date +"TBlock_%Y-%m-%d_%H-%M-%S.txt")"
wait
taskset -c 0 ./Matrix -size $size -n $n -sync spin > "$(date +"TSpin_%Y-%m-%d_%H-%M-%S.txt")"
wait


gnuplot -persist <<-EOFMarker
    set title "Time vs Size without sysbench" 
    
    set datafile separator "|"
    
    plot "TSpin.txt" with lines, "Tblock.txt" with lines
EOFMarker