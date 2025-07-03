ps_2_0
def c2, -0.500000, 1.000000, 0.000000, 0.000000
def c3, 0.700000, 0.600000, 0.500000, 0.000000
dcl t1.rg
dcl t5.rgb
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
texld r1, t1, s1
add r0, r1.a, c2.r
mul r1.rgb, r1, c3
rcp r1.a, t7.a
mul r2.rg, r1.a, t7
mad r2.rg, r2, c0, c0.abgr
texkill r0
texld r0, r2, s0
dp3 r2.r, t5, t5
add r1.a, -r2.r, c2.g
max r0.a, r1.a, c2.b
mul r0.a, r0.a, r0.a
mul r1.rgb, r1, r0.a
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c1
mov r0.a, c2.b
mov oC0, r0
