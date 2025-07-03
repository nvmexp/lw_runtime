ps_2_0
def c3, 2.000000, -1.000000, -0.500000, 0.000000
def c4, 15.000000, 1.000000, 0.000000, 0.000000
dcl t1.rg
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld r1, t1, s0
add r0, r1.a, c3.b
mad r4.rgb, c3.r, r1, c3.g
rcp r1.a, t7.a
mul r1.rg, r1.a, t7
mad r2.rg, r1, c0, c0.abgr
texkill r0
texld r1, t1, s1
texld r0, r2, s2
dp3 r2.r, t6, t6
rsq r0.a, r2.r
mul r3.rgb, r0.a, t6
nrm r2.rgb, r4
dp3 r3.r, r2, r3
mul r3.rgb, r2, r3.r
add r3.rgb, r3, r3
mad r4.rgb, t6, -r0.a, r3
nrm r3.rgb, t4
dp3 r4.r, r4, r3
max r2.a, r4.r, c3.a
mul r2.rgb, r2, c1.r
pow r0.a, r2.a, c4.r
dp3 r2.r, r2, r3
mul r2.a, r1.a, r0.a
max r1.a, r2.r, c3.a
dp3 r2.r, t5, t5
add r3.a, -r2.r, c4.g
max r0.a, r3.a, c3.a
mad r1.rgb, r1, r1.a, r2.a
mul r0.a, r0.a, r0.a
mul r1.rgb, r1, r0.a
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c2
mov r0.a, c3.a
mov oC0, r0
