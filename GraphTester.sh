#!/bin/bash
size=500
n=5

taskset -c 0 ./Matrix -size $size -n $n -sync block >block.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
wait
taskset -c 0 ./Matrix -size $size -n $n -sync spin >spin.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
wait
taskset -c 0 ./Matrix -size $size -n $n -sync auto >auto.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 
wait
taskset -c 0 ./Matrix -size $size -n $n -sync yield >yield.txt &  taskset -c 0 sysbench --test=cpu --cpu-max-prime=20000 --max-time=30 run 

gnuplot -persist <<-EOFMarker
    set title "Time vs Size with sysbench" 
    
    set datafile separator ":"
    
    plot "spin.txt" using 3:1 with lines,"block.txt" using 3:1 with lines
EOFMarker


