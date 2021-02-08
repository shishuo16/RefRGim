#!/bin/bash

### 1=> 1KG reference panel path
### 2=> RefRGim path
### 3=> output prefix

echo -e "==>Reconstructing study specified reference panel (chr1-22)..."
for((i=1;i<=22;i++));  
do   
    while [ 1 ]; do
        num=`ps -ef|grep RefRGim_reconstruct.sh|wc -l`
        if [ $num -lt 6 ]; then
            sh $2/scripts/RefRGim_reconstruct.sh $1 $i $3 &
            break
        fi
        sleep 100s
    done
done  

while [ 1 ]; do
    num=`ps -ef|grep RefRGim_reconstruct.sh|wc -l`
    if [ $num -eq 1 ]; then
        echo -e "\n==>RefRGim is done. Having a nice day."
        break
    fi
done
