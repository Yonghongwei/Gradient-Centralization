#cifar100 e200 bs128  gs  2,4,8,16
import os,time

#print('runing mini_imagenet.py')


os.system("nohup  python -W ignore main.py /home/yonghw/data/mini_imagenet/split_mini/ --model r50  -b 128 --alg sgd   > logout/r50_b128_sgd.log  ")

os.system("nohup  python -W ignore main.py /home/yonghw/data/mini_imagenet/split_mini/ --model r50  -b 128 --alg sgdGC   > logout/r50_b128_sgdGC.log  ")

os.system("nohup  python -W ignore main.py /home/yonghw/data/mini_imagenet/split_mini/ --model r50ws  -b 128 --alg sgd   > logout/r50ws_b128_sgd.log  ")

os.system("nohup  python -W ignore main.py /home/yonghw/data/mini_imagenet/split_mini/ --model r50ws  -b 128 --alg sgdGC   > logout/r50ws_b128_sgdGC.log  ")
