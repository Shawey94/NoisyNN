#! /bin/bash
dir=/home/minheng/UTA/NoisyNN/NoisyViT-OptimalQ/ImageNet1K/train

for file in `ls $dir/*tar` 
do

filename=`basename $file .tar`
echo $file
mkdir $dir/$filename
tar -xvf $file -C $dir/$filename
rm $file

done