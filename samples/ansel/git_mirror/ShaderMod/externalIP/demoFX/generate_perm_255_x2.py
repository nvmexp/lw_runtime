import random

a = [i for i in range(256)]
random.shuffle(a)

for i in range(2):
    for j in range(0, 256, 4):
        print("{"+str(a[j]) + "," + str(a[j+1]) + "," + str(a[j+2])+ "," + str(a[j+3])+"},")
