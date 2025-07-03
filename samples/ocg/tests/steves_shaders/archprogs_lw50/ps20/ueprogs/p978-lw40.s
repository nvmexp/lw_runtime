ps_2_0
def c2, 2.000000, -1.000000, 10.000000, 0.000000
def c3, 15.000000, 0.000000, 0.000000, 0.000000
dcl t1.rg
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
texld r0, t1, s1
mad r4.rgb, c2.r, r0, c2.g
mul r1.rg, t1, c2.b
rcp r0.a, t7.a
mul r0.rg, r0.a, t7
mad r0.rg, r0, c0, c0.abgr
texld r3, r1, s0
texld r2, t1, s3
texld r1, t1, s2
texld r0, r0, s4
mad r3.rgb, c2.r, r3, r4
add r4.rgb, r3, c2.g
dp3 r5.r, t6, t6
nrm r3.rgb, r4
rsq r0.a, r5.r
mul r4.rgb, r0.a, t6
dp3 r4.r, r3, r4
mul r4.rgb, r3, r4.r
add r4.rgb, r4, r4
mad r5.rgb, t6, -r0.a, r4
nrm r4.rgb, t5
dp3 r5.r, r5, r4
max r1.a, r5.r, c2.a
dp3 r3.r, r3, r4
pow r0.a, r1.a, c3.r
mul r2.rgb, r2, r0.a
max r0.a, r3.r, c2.a
mad r1.rgb, r1, r0.a, r2
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c1
mov r0.a, c2.a
mov oC0, r0
