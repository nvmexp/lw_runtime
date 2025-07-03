ps_2_0
def c2, -0.500000, 0.000000, 0.000000, 0.000000
def c3, 0.700000, 0.600000, 0.500000, 0.000000
dcl t1.rg
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
texld r0, t1, s1
add r1, r0.a, c2.r
rcp r0.a, t7.a
mul r2.rg, r0.a, t7
mad r2.rg, r2, c0, c0.abgr
texkill r1
texld r1, r2, s0
mul r1.rgb, r1, v0.r
mul_pp r0.rgb, r0, r1
mul r0.rgb, r0, c1
mul r0.rgb, r0, c3
mov r0.a, c2.g
mov oC0, r0
