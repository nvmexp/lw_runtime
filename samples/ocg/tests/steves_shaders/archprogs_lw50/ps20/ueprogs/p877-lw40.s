ps_2_0
def c1, 2.000000, -1.000000, -0.500000, 0.500000
dcl t1.rg
dcl t5
dcl_pp t7.rgb
dcl_2d s0
dcl_2d s1
texld r0, t1, s1
add r1, r0.a, c1.b
texld r2, t1, s0
mad_pp r3.rgb, c1.r, r2, c1.g
nrm_pp r2.rgb, r3
nrm_pp r3.rgb, t7
dp3 r2.r, r2, r3
mad r0.a, r2.r, c1.a, c1.a
texkill r1
mul r0.a, r0.a, r0.a
mul_pp r0.rgb, r0, r0.a
mul_pp r0.rgb, r0, c0
mov r0.a, t5.a
mov oC0, r0
