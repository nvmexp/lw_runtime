for k in [8, 16, 32]:
    for modeA in ["m,k,c,n", "k,m,c,n"]:
        m = 256
        while m < 8*1024:
            print("./lwtensorTest -algo-101 -Rcontraction -modeA%s -modeBa,c,k,b -modeCm,n,a,b -extentm=%d,n=%d,k=%d,c=2,a=2,b=2 -fastVerify -beta0"%(modeA,m/2,m/2,k/2))
            if (m < 2048):
                m += 128
            elif (m < 4*1024):
                m += 512
            else:
                m += 1024
