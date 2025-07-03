./lwtensorTest -Rcontraction -Pas -Pbs -Pcs -Pcomps -modeAk,a,b -modeBn,k,x -modeCn,x,a,b -extenta=4096,b=1,k=4096,n=4096,x=2 -alpha1 -beta0 -fastVerify
./lwtensorTest -Rcontraction -Pas -Pbs -Pcs -Pcomps -modeAk,a,b -modeBk,x,n -modeCn,x,a,b -extenta=32,b=1,k=32,n=32,x=1 -alpha1 -beta0
./lwtensorTest -Rcontraction -Pas -Pbs -Pcs -Pcomps -algo-4 -modeAm,k,a -modeBn,k,b -modeCm,n,a,b -extentk=17,m=288,n=288,a=2,b=3 -beta10 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAa,k -modeBk,n -modeCa,n -extenta=8,k=8,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAa,k -modeBn,k -modeCa,n -extenta=8,k=8,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a -modeBk,n -modeCa,n -extenta=8,k=8,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a -modeBn,k -modeCa,n -extenta=8,k=8,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAa,k -modeBk,n -modeCa,n -extenta=80,k=8,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAa,k -modeBn,k -modeCa,n -extenta=8,k=80,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a -modeBk,n -modeCa,n -extenta=8,k=8,n=80 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a -modeBn,k -modeCa,n -extenta=320,k=80,n=320 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAa,k -modeBk,n -modeCa,n -extenta=1000,k=320,n=320 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAm,k,a -modeBn,k,b -modeCm,n,a,b -extentk=17,m=288,n=288,a=2,b=3 -beta10 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAm,k,a -modeBn,k -modeCm,a,n -extenta=80,m=8,k=80,n=72 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAm,k,a -modeBn,b,k -modeCm,a,b,n -extenta=80,b=2,m=8,k=80,n=320 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a,m -modeBn,b,k -modeCm,a,b,n -extenta=80,b=2,m=8,k=80,n=320 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a,m -modeBk,n,b -modeCm,a,b,n -extenta=80,b=2,m=8,k=80,n=320 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a,c,m -modeBk,n,c,b -modeCm,a,b,n -extenta=80,c=3,b=2,m=8,k=80,n=320 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,c,m -modeBk,n,c -modeCm,n -extentc=3,m=8,k=80,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,c,m -modeBn,c,k -modeCm,n -extentc=3,m=8,k=80,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a,c,m -modeBn,c,b,k -modeCm,a,n,b -extentc=3,a=4,b=7,m=8,k=80,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a,c,m -modeBn,c,b,k -modeCm,a,b,n -extentc=3,a=4,b=7,m=8,k=80,n=8 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a,l,c,m -modeBn,c,b,k,l -modeCm,a,n,l,b -extentc=3,a=4,b=7,m=8,k=80,n=8,l=3 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,a,l,c,y,m -modeBn,c,b,y,k,l -modeCm,a,n,l,b,y -extentc=3,a=4,b=7,m=8,k=80,n=8,l=3,y=2 -alpha1 -beta0 -numRuns3 
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAk,c,a,b -modeBk,c,n -modeCn,a,b -extenta=8192,k=6400,c=2,b=2,n=16384 -alpha1 -beta0 -numRuns3 -fastVerify
./lwtensorTest -Rcontraction -Pah -Pbh -Pch -Pcomps -algo-4 -modeAa,b,k,c, -modeBn,k,c -modeCn,a,b -extenta=8192,k=6400,c=1,b=1,n=16384 -alpha1 -beta0 -numRuns3 -fastVerify
