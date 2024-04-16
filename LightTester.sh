#!/bin/bash
size=1000
n=5



#taskset -c 0 ./Matrix -size $size -cycles $cycles -n $n -sync spin > "$(date +"TBlock_%Y-%m-%d_%H-%M-%S.txt")"
#wait
#taskset -c 0 ./Matrix -size $size -cycles $cycles -n $n -sync spin > "$(date +"TSpin_%Y-%m-%d_%H-%M-%S.txt")"
#wait
for cycles in {1..50}; do
  
  timestamp=$(date +"%Y-%m-%d_%H-%M")
  mkdir -p "TBlock_$timestamp" "TSpin_$timestamp"


  taskset -c 0 ./Matrix -size $size -cycles $cycles -n $n -sync block > "TBlock_$timestamp/$(date +"TBlock_%H-%M-%S.txt")"
  wait


  taskset -c 0 ./Matrix -size $size -cycles $cycles -n $n -sync spin > "TSpin_$timestamp/$(date +"TSpin_%H-%M-%S.txt")"
  wait
done

gnuplot -persist <<-EOFMarker
    set title "Time vs Size without sysbench" 
    
    set datafile separator "|"
    
    plot "TSpin.txt" with lines, "Tblock.txt" with lines
EOFMarker