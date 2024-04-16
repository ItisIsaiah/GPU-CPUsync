#!/bin/bash
size=1000
n=5


for cycles in {1..50}; do
 
  timestamp=$(date +"%Y-%m-%d_%H-%M")
  mkdir -p "Block_$timestamp" "Spin_$timestamp"


  taskset -c 0 ./Matrix -size $size -cycles $cycles -n $n -sync block > "Block_$timestamp/$(date +"Block_%H-%M-%S.txt")" & taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
  wait

  taskset -c 0 ./Matrix -size $size -cycles $cycles -n $n -sync spin > "Spin_$timestamp/$(date +"Spin_%H-%M-%S.txt")" & taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
  wait
done

#taskset -c 0 ./Matrix -size $size -n $n -sync block >block.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
#wait
#taskset -c 0 ./Matrix -size $size -n $n -sync spin >spin.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
#wait
#taskset -c 0 ./Matrix -size $size -n $n -sync auto >auto.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
#wait
#taskset -c 0 ./Matrix -size $size -n $n -sync yield >yield.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 

gnuplot -persist <<-EOFMarker
    set title "Time vs Size with sysbench" 
    
    set datafile separator ":"
    
    plot "spin.txt" using 3:1 with lines,"block.txt" using 3:1 with lines
EOFMarker


