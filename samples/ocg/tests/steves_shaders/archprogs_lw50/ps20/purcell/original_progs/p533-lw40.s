ps_2_0
def c0, 0.333330, 0.000000, 0.000000, 0.000000
dcl t0.rg
dcl t1.rg
dcl_pp t2.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
texld r2, t2, s3
texld r3, t0, s0
texld r4, t1, s1
texld r0, t2, s2
dp3 r3.a, r3, c0.r
dp3 r4.a, r4, c0.r
lrp r1, r2.a, r3, r4
add r0, r0, r0
mul r0, r1, r0
mov oC0, r0
