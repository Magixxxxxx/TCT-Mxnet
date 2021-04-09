#!/bin/bash
# 1. mkdir directory garbage
if [ -d ./garbage ]
then
rm -rf ./garbage
mkdir garbage
else
mkdir garbage
fi

# 2. create annotation_file.txt and shuffle
find /root/commonfile/TCTAnnotatedData/TCTAnnotated20210331 -name "*.txt" > annotation_file.txt
# sed -i 's/^...//g' annotation_file.txt
cat annotation_file.txt | sort -R > annotation_file_tp.txt
rm annotation_file.txt
mv annotation_file_tp.txt annotation_file.txt

# 3. create train.txt:7, test.txt:2, val.txt:1
count=`cat annotation_file.txt | wc -l`
node1=`expr $count \* 7 / 10`
node2=`expr $count \* 9 / 10`
sed -n "1,${node1}p" annotation_file.txt > train.txt
node_tp=`expr ${node1} + 1`
sed -n "${node_tp},${node2}p" annotation_file.txt > test.txt
node_tp=`expr ${node2} + 1`
sed -n "${node_tp},\$p" annotation_file.txt > val.txt

if [ -d ./json ]
then
rm -rf ./json
mkdir json
else
mkdir json
fi

# create json file
python labelme2coco.py val.txt val.json 
python labelme2coco.py test.txt test.json 
python labelme2coco.py train.txt train.json 

# mv temp file to garbage 
mv *.txt ./garbage
