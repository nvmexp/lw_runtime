import random
random.seed(0)

targetCTAs = range(40,161,2)
tests = []
for m in [32,64,128,256]:
    for n in [32,64,128,256]:
        for modeA in ["m,k,c", "k,m,c"]:
            k = 20*max(m,n)
            while k < 100*1024:
                if (m*n*k >= 700**3 and m*n*k <= 6144**3):
                    tests.append("./lwtensorTest -algo-101 -Rcontraction -modeA%s -modeBa,c,k -modeCm,a -extentm=%d,a=%d,k=%d,c=2 -fastVerify -beta0"%(modeA,m,n,k/2))
                k += random.choice([1024,2048,4096,6144])

numGPUs = 8
maxNumCases = 256
random.shuffle(tests)

for g in range(numGPUs):
    workload = maxNumCases  / numGPUs
    start = g * workload
    end = min(maxNumCases, (g+1) * workload)
    if (start == end):
        break
    f = open("gemm_large_k_%d.sh"%g, "w+")
    for i in range(start, end):
        for targetCTA in targetCTAs:
            f.write(tests[i] + " -partitionsK%d\n"%targetCTA)
