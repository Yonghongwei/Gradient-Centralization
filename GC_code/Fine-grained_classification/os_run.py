
import os,time




os.system("nohup python -W ignore main.py /home/yonghw/data/data/CUB_200_2011/ --model r50p -b 128 --alg sgd --dataset cub  > logout/Cub_r50p_sgd_b128_g4.log ")
os.system("nohup python -W ignore main.py /home/yonghw/data/data/CUB_200_2011/ --model r50p -b 128  --alg sgdGC --dataset cub > logout/Cub_r50p_sgdGC_b128_g4.log ")

os.system("nohup python -W ignore main.py /home/yonghw/data/data/Car196/ --model r50p -b 128 --alg sgd --dataset cars > logout/Car_r50p_sgd_b128_g4.log ")
os.system("nohup python -W ignore main.py /home/yonghw/data/data/Car196/ --model r50p -b 128 --alg sgdGC --dataset cars> logout/Car_r50p_sgdGC_b128_g4.log ")

os.system("nohup python -W ignore main.py /home/yonghw/data/data/fgvc_aricraft/ --model r50p  -b 128 --alg sgd --dataset fgvc > logout/Ari_r50p_sgd_b128_g4.log ")
os.system("nohup python -W ignore main.py /home/yonghw/data/data/fgvc_aricraft/ --model r50p  -b 128 --alg sgdGC --dataset fgvc > logout/Ari_r50p_sgdGC_b128_g4.log ")

os.system("nohup python -W ignore main.py /home/yonghw/data/data/StanfordDogs/ --model r50p  -b 128  --alg sgd --dataset dogs > logout/Dog_r50p_sgd_b128_g4.log ")
os.system("nohup python -W ignore main.py /home/yonghw/data/data/StanfordDogs/ --model r50p  -b 128  --alg sgdGC --dataset dogs > logout/Dog_r50p_sgdGC_b128_g4.log ")
