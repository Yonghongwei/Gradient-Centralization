
import os,time

#cifar100 sgd & sgdGCC

os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg sgd --epochs 200  --model r50 > logout/r50_lr11_wd45_sgd.log ")

os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg sgdGC --epochs 200  --model r50 > logout/r50_lr11_wd45_sgdGC.log ")
