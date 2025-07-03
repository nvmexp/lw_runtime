ps_2_0
def c1, 0.020000, -0.008000, 2.000000, -1.000000
def c2, -0.500000, 0.500000, 0.000000, 0.000000
dcl t1.rg
dcl t5
dcl t6.rgb
dcl_pp t7.rgb
dcl_2d s0
dcl_2d s1
texld r0, t1, s0
dp3 r0.r, t6, t6
rsq r1.a, r0.r
mul_pp r0.rg, r1.a, t6
mad r0.a, r0.a, c1.r, c1.g
mad r2.rg, r0, r0.a, t1
texld r0, r2, s1
add r1, r0.a, c2.r
texld r2, r2, s0
mad_pp r3.rgb, c1.b, r2, c1.a
nrm_pp r2.rgb, r3
nrm_pp r3.rgb, t7
dp3 r2.r, r2, r3
mad r0.a, r2.r, c2.g, c2.g
texkill r1
mul r0.a, r0.a, r0.a
mul_pp r0.rgb, r0, r0.a
mul_pp r0.rgb, r0, c0
mov r0.a, t5.a
mov oC0, r0
