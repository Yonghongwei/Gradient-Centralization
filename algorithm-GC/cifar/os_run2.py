#cifar100 e200 bs128  gs  2,4,8,16
import os,time


#r50
##############


### adam 
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr21_wd45_adam_1.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr21_wd45_adam_2.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr21_wd45_adam_3.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr21_wd45_adam_4.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr21_wd45_adam_5.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr21_wd45_adam_6.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr21_wd45_adam_7.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr21_wd45_adam_8.log ")

time.sleep(500)
### adamGC 
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr21_wd45_adamGC_1.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr21_wd45_adamGC_2.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr21_wd45_adamGC_3.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr21_wd45_adamGC_4.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr21_wd45_adamGC_5.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr21_wd45_adamGC_6.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr21_wd45_adamGC_7.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr21_wd45_adamGC_8.log ")
time.sleep(500)

### adamGCC 
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr21_wd45_adamGCC_1.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr21_wd45_adamGCC_2.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr21_wd45_adamGCC_3.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr21_wd45_adamGCC_4.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr21_wd45_adamGCC_5.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr21_wd45_adamGCC_6.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr21_wd45_adamGCC_7.log &")
os.system("nohup  python  main.py --lr 0.01 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr21_wd45_adamGCC_8.log ")
time.sleep(500)

##############
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr25_wd45_adam_1.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr25_wd45_adam_2.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr25_wd45_adam_3.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr25_wd45_adam_4.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr25_wd45_adam_5.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr25_wd45_adam_6.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr25_wd45_adam_7.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr25_wd45_adam_8.log ")

time.sleep(500)
### adamGC 
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr25_wd45_adamGC_1.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr25_wd45_adamGC_2.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr25_wd45_adamGC_3.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr25_wd45_adamGC_4.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr25_wd45_adamGC_5.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr25_wd45_adamGC_6.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr25_wd45_adamGC_7.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr25_wd45_adamGC_8.log ")
time.sleep(500)

### adamGCC 
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr25_wd45_adamGCC_1.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr25_wd45_adamGCC_2.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr25_wd45_adamGCC_3.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr25_wd45_adamGCC_4.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr25_wd45_adamGCC_5.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr25_wd45_adamGCC_6.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr25_wd45_adamGCC_7.log &")
os.system("nohup  python  main.py --lr 0.05 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr25_wd45_adamGCC_8.log ")
time.sleep(500)



##############
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr115_wd45_adam_1.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr115_wd45_adam_2.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr115_wd45_adam_3.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr115_wd45_adam_4.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr115_wd45_adam_5.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr115_wd45_adam_6.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr115_wd45_adam_7.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adam   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr115_wd45_adam_8.log ")

time.sleep(500)
### adamGC 
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr115_wd45_adamGC_1.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr115_wd45_adamGC_2.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr115_wd45_adamGC_3.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr115_wd45_adamGC_4.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr115_wd45_adamGC_5.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr115_wd45_adamGC_6.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr115_wd45_adamGC_7.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGC   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr115_wd45_adamGC_8.log ")
time.sleep(500)

### adamGCC 
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 10 > logout2/r50_lr115_wd45_adamGCC_1.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 11 > logout2/r50_lr115_wd45_adamGCC_2.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 12 > logout2/r50_lr115_wd45_adamGCC_3.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 13 > logout2/r50_lr115_wd45_adamGCC_4.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 14 > logout2/r50_lr115_wd45_adamGCC_5.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 15 > logout2/r50_lr115_wd45_adamGCC_6.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 16 > logout2/r50_lr115_wd45_adamGCC_7.log &")
os.system("nohup  python  main.py --lr 0.15 --wd 0.0005 --alg adamGCC   --epochs 200  --model r50 --gpug 17 > logout2/r50_lr115_wd45_adamGCC_8.log ")
time.sleep(500)




#
###############
###############
#
#### adamW 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_adamW_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_adamW_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_adamW_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_adamW_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_adamW_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_adamW_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_adamW_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamW   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_adamW_8.log ")
#
#time.sleep(500)
#### adamWGC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_adamWGC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_adamWGC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_adamWGC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_adamWGC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_adamWGC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_adamWGC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_adamWGC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_adamWGC_8.log ")
#time.sleep(500)

### adamWGCC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_adamWGCC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_adamWGCC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_adamWGCC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_adamWGCC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_adamWGCC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_adamWGCC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_adamWGCC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg adamWGCC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_adamWGCC_8.log ")
#time.sleep(500)
#
###############
###############
#
#### radam 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_radam_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_radam_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_radam_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_radam_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_radam_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_radam_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_radam_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radam   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_radam_8.log ")
#
#time.sleep(500)
#### radamGC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_radamGC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_radamGC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_radamGC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_radamGC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_radamGC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_radamGC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_radamGC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_radamGC_8.log ")
#time.sleep(500)
#
#### radamGCC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_radamGCC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_radamGCC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_radamGCC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_radamGCC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_radamGCC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_radamGCC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_radamGCC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg radamGCC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_radamGCC_8.log ")
#time.sleep(500)
#
###############
###############
#
#### Lsgd 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_Lsgd_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_Lsgd_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_Lsgd_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_Lsgd_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_Lsgd_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_Lsgd_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_Lsgd_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Lsgd   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_Lsgd_8.log ")
#time.sleep(500)
#
#### LsgdGC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_LsgdGC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_LsgdGC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_LsgdGC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_LsgdGC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_LsgdGC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_LsgdGC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_LsgdGC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_LsgdGC_8.log ")
#time.sleep(500)
#
#### LsgdGCC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_LsgdGCC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_LsgdGCC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_LsgdGCC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_LsgdGCC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_LsgdGCC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_LsgdGCC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_LsgdGCC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LsgdGCC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_LsgdGCC_8.log ")
#time.sleep(500)
#
###############
###############
#
#### Ladam 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_Ladam_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_Ladam_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_Ladam_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_Ladam_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_Ladam_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_Ladam_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_Ladam_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg Ladam   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_Ladam_8.log ")
#
#time.sleep(500)
#### LadamGC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_LadamGC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_LadamGC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_LadamGC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_LadamGC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_LadamGC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_LadamGC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_LadamGC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_LadamGC_8.log ")
#time.sleep(500)
#
#### LadamGCC 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_LadamGCC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_LadamGCC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_LadamGCC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_LadamGCC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_LadamGCC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_LadamGCC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_LadamGCC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg LadamGCC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_LadamGCC_8.log ")
#time.sleep(500)
#
###############
###############
#
#### ranger
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_ranger_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_ranger_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_ranger_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_ranger_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_ranger_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_ranger_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_ranger_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg ranger   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_ranger_8.log ")
#
#time.sleep(500)
#### ranger 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_rangerGC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_rangerGC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_rangerGC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_rangerGC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_rangerGC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_rangerGC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_rangerGC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_rangerGC_8.log ")
#time.sleep(500)
#
#### ranger 
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 10 > logout/r50_lr11_wd45_rangerGCC_1.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 11 > logout/r50_lr11_wd45_rangerGCC_2.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 12 > logout/r50_lr11_wd45_rangerGCC_3.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 13 > logout/r50_lr11_wd45_rangerGCC_4.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 14 > logout/r50_lr11_wd45_rangerGCC_5.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 15 > logout/r50_lr11_wd45_rangerGCC_6.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 16 > logout/r50_lr11_wd45_rangerGCC_7.log &")
#os.system("nohup  python  main.py --lr 0.1 --wd 0.0005 --alg rangerGCC   --epochs 200  --model r50 --gpug 17 > logout/r50_lr11_wd45_rangerGCC_8.log ")
#time.sleep(500)

##############


