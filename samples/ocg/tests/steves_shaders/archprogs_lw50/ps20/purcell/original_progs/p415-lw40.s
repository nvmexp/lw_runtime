ps_2_0
def c0, 2.000000, -1.000000, 1.000000, 0.000000
dcl t0.rg
dcl t1.rgb
dcl t2.rgb
dcl t3.rg
dcl t4.rg
dcl t5.rg
dcl t6.rg
dcl t7.rg
dcl_2d s2
dcl_2d s3
dcl_2d s4
texld r0, t0, s3
nrm r1.rgb, t1
mad r0.rgb, c0.r, r0, c0.g
dp3_pp r1.r, r1, r0
add r1.a, -r1.r, c0.b
mul r2.a, r1.a, r1.a
mul r2.a, r2.a, r2.a
mul r3.a, r1.a, r2.a
mov r1.rg, t2
dp2add r4.a, t3, r0, r1.r
rcp r2.a, t2.b
dp2add r1.a, t4, r0, r1.g
mul r1.r, r4.a, r2.a
mul r1.g, r2.a, r1.a
mov r2.rg, t5
dp2add r4.a, t6, r0, r2.r
dp2add r1.a, t7, r0, r2.g
mul r0.r, r2.a, r4.a
mul r0.g, r2.a, r1.a
texld r2, r1, s4
texld r1, r0, s2
mul r0.rgb, r2, c4
mul r1.rgb, r1, c1
mad r0.rgb, r0, r3.a, r1
mov oC0, r0
