#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0"  )" && pwd  )"
cd $THIS_DIR

pip3 install -i https://pypi.douban.com/simple/ timm
cd datasets
mkdir coco
cd coco
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/COCO/train2017.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/COCO/val2017.zip
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/COCO/test2017.zip
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
mkdir annotations
cd ./annotations
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/PoseLabel/person_keypoints_train2017.json ./
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/PoseLabel/person_keypoints_val2017.json ./
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/PoseLabel/instances_val2017.json ./
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/PoseLabel/instances_train2017.json ./
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/datasets/PoseLabel/image_info_test-dev2017.json

cd $THIS_DIR

