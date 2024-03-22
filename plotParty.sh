#!/bin/bash
gnuplot -persist <<-EOFMarker
    set title "Time vs Size without sysbench" 
    
    set datafile separator "|"
    
    plot "TSpin.txt" with lines, "Tblock.txt" with lines
EOFMarker