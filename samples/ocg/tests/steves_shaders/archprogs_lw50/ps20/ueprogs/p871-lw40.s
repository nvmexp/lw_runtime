ps_2_0
def c1, 0.500000, 0.000000, 0.000000, 0.000000
dcl t1.rg
dcl t5
dcl_pp t7.rgb
dcl_2d s0
texld_pp r0, t1, s0
dp3_pp r1.r, t7, t7
rsq_pp r0.a, r1.r
mul_pp r0.a, r0.a, t7.b
mad r0.a, r0.a, c1.r, c1.r
mul r0.a, r0.a, r0.a
mul_pp r0.rgb, r0, r0.a
mul_pp r0.rgb, r0, c0
mov r0.a, t5.a
mov oC0, r0
