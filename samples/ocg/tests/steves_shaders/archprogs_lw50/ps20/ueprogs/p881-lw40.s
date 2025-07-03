ps_2_0
def c6, 0.100000, 1.000000, 0.000000, 0.000000
dcl t7
dcl_2d s0
dcl_2d s1
rcp r0.a, t7.a
mul r1.rg, r0.a, t7
mad r0.rg, r1, c4, c4.abgr
texld r0, r0, s0
mul r0.rg, r1, r0.a
mul r1.rg, r0.g, c1
mad r0.rg, c0, r0.r, r1
mad r0.rg, c2, r0.a, r0
add r0.rg, r0, c3
mov r0.a, c6.r
mad r0.rg, r0, r0.a, c5
texld_pp r0, r0, s1
mov r0.a, c6.g
mov oC0, r0
