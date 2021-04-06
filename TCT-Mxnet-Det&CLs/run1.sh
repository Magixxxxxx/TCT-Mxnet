<<<<<<< HEAD
python3 fasterrcnn_cls_norm.py --num-workers 4 --gpus 0,1,2,3 --batch-size 4 \
--epochs 20 --lr 0.002 --lr-decay-epoch 10,15 \
--dataset coco --network resnet50_v1b --val-interval 1 --use-fpn --details TCT
=======
python3 fasterrcnn_cls_norm.py --num-workers 4 --gpus 0,1,2,3,4,5,6,7 --batch-size 8 --dataset coco --network resnet50_v1b --epochs 12 --lr 0.003 --val-interval 1 --use-fpn --details TCT --lr-decay-epoch 8,10
>>>>>>> 76c6bc30b1718a866a79b0d17e85473ae1ab8b76
