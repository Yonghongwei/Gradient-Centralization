
import os,time


os.system("#nohup python -W ignore main.py /ssd/data/yonghw/Imagenet/v0/ --model r50bn --alg sgd -b 256 --path r50bn_sgd_b256_g4 > logout/r50bn_sgd_b256_g4.log &")

os.system("#nohup python -W ignore main.py /ssd/data/yonghw/Imagenet/v0/ --model r50bn --alg sgdGC -b 256 --path r50bn_sgdGC_b256_g4 > logout/r50bn_sgdGC_b256_g4.log &")
