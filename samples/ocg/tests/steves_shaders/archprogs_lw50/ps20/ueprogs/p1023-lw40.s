ps_2_0
def c3, 0.000000, 0.000000, 2.000000, -0.500000
def c4, 0.350000, 0.500000, 0.300000, 0.000000
def c5, 0.000000, 0.000000, 1.000000, 15.000000
dcl t1.rg
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
texld r1, t1, s1
add r0, r1.a, c3.a
mul r1.rgb, r1, c4
rcp r1.a, t7.a
mul r2.rg, r1.a, t7
mad r2.rg, r2, c0, c0.abgr
texkill r0
texld r0, r2, s0
nrm r2.rgb, t6
mad r2.rgb, r2.b, c3, -r2
nrm r3.rgb, t4
dp3 r2.r, r2, r3
max r1.a, r2.r, c3.r
mov r0.a, c1.r
mul r2.rgb, r0.a, c5
pow r0.a, r1.a, c5.a
dp3 r2.r, r2, r3
max r1.a, r2.r, c3.r
dp3 r2.r, t5, t5
add r2.a, -r2.r, c5.b
mul r2.rgb, r1, r0.a
max r0.a, r2.a, c3.r
mad r1.rgb, r1, r1.a, r2
mul r0.a, r0.a, r0.a
mul r1.rgb, r1, r0.a
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c2
mov r0.a, c3.r
mov oC0, r0
