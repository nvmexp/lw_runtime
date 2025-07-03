ps_2_0
def c1, 2.000000, -1.000000, 0.500000, 0.000000
dcl t1.rg
dcl t5
dcl_pp t7.rgb
dcl_2d s0
dcl_2d s1
texld r1, t1, s0
texld_pp r0, t1, s1
mad_pp r2.rgb, c1.r, r1, c1.g
nrm_pp r1.rgb, r2
nrm_pp r2.rgb, t7
dp3 r1.r, r1, r2
mad r0.a, r1.r, c1.b, c1.b
mul r0.a, r0.a, r0.a
mul_pp r0.rgb, r0, r0.a
mul_pp r0.rgb, r0, c0
mov r0.a, t5.a
mov oC0, r0
