lwtensorTest -modeC0,a,1,b,2,c,3,d,4,e,5,f -modeA0,2,1,3,4,5,x -modeBa,c,b,d,e,f,x -extent0=4,1=4,2=4,3=4,4=4,5=4,a=4,b=4,c=4,d=4,e=4,f=4,x=256, -Pac -Pbc -Pcc -Pcomps -Rcontraction -algo-101
lwtensorTest -modeAa,k,b -modeCb,a,n,m -modeBm,k,n -extentk=8,m=128,n=128,a=128,b=128 -Rcontraction -fastVerify
lwtensorTest -modeCm,a,n,b -modeAm,k,n,c -modeBa,k,b,c -extentm=32,n=2,a=32,b=2,k=32,c=2048, -Pac -Pbc -Pcc -Pcomps -Rcontraction -algo-101
