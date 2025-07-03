ps_2_0
def c2, 2.000000, -1.000000, 0.000000, 15.000000
dcl t0.rg
dcl t1.rg
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
texld r2, t1, s0
texld r1, t1, s2
texld r0, t1, s1
dp3 r3.r, t6, t6
rsq r1.a, r3.r
mul r3.rgb, r1.a, t6
mad r4.rgb, c2.r, r2, c2.g
nrm r2.rgb, r4
dp3 r3.r, r2, r3
mul r3.rgb, r2, r3.r
add r3.rgb, r3, r3
mad r4.rgb, t6, -r1.a, r3
nrm r3.rgb, t5
dp3 r4.r, r4, r3
dp3 r2.r, r2, r3
max r2.a, r4.r, c2.b
pow r1.a, r2.a, c2.a
mul r1.rgb, r1, r0.a
mul r1.rgb, r1.a, r1
max r0.a, r2.r, c2.b
mad r2.rgb, r0, r0.a, r1
rcp r0.a, t7.a
mul r0.rg, r0.a, t7
mad r0.rg, r0, c0, c0.abgr
texld r0, r0, s3
texld r1, t0, s4
mul r0.rgb, r0, r1.r
mul r0.rgb, r2, r0
mul r0.rgb, r0, c1
mov r0.a, c2.b
mov oC0, r0
