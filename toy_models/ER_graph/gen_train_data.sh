#!/bin/bash

rm -f output.txt
touch output.txt

for i in `seq 100 5 1000`
do
    l=$(echo "1/$i" | bc -l)
    r=$(echo "l($i)/$i" | bc -l)
    inc=$(echo "( $r-$l )/10" | bc -l)
    echo $i $l $r $inc
    for j in `seq $l $inc $r`
    do
        echo -n "$i $j " >>output.txt
        ./er $i 100 $j >>output.txt
    done
done
