ps_2_0
def c2, 2.000000, -1.000000, 0.050000, -0.015000
def c3, 0.000000, 15.000000, 1.000000, 0.000000
dcl t1.rg
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld r0, t1, s0
mad r0.a, c2.r, r0.a, c2.g
dp3 r0.r, t6, t6
rsq r3.a, r0.r
mad r0.a, r0.a, c2.b, c2.a
mul r3.rgb, r3.a, t6
mad r1.rg, r3, r0.a, t1
rcp r0.a, t7.a
mul r0.rg, r0.a, t7
mad r0.rg, r0, c0, c0.abgr
texld r2, r1, s0
texld r1, r1, s1
texld r0, r0, s2
mad r4.rgb, c2.r, r2, c2.g
nrm r2.rgb, r4
dp3 r3.r, r2, r3
mul r3.rgb, r2, r3.r
add r3.rgb, r3, r3
mad r4.rgb, t6, -r3.a, r3
nrm r3.rgb, t4
dp3 r4.r, r4, r3
max r1.a, r4.r, c3.r
dp3 r2.r, r2, r3
pow r0.a, r1.a, c3.g
max r1.a, r2.r, c3.r
dp3 r2.r, t5, t5
add r2.a, -r2.r, c3.b
mul r2.rgb, r1, r0.a
max r0.a, r2.a, c3.r
mad r1.rgb, r1, r1.a, r2
mul r0.a, r0.a, r0.a
mul r1.rgb, r1, r0.a
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c1
mov r0.a, c3.r
mov oC0, r0
