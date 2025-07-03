ps_2_0
def c1, 0.050000, -0.025000, 2.000000, -1.000000
def c2, 0.500000, 0.000000, 0.000000, 0.000000
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
mad r1.a, r0.a, c1.r, c1.g
mad r0.a, r0.a, c1.r, c1.g
mad r1.rg, r0, r1.a, t1
mad r0.rg, r0, r0.a, t1
texld r1, r1, s0
texld_pp r0, r0, s1
mad_pp r2.rgb, c1.b, r1, c1.a
nrm_pp r1.rgb, r2
nrm_pp r2.rgb, t7
dp3 r1.r, r1, r2
mad r0.a, r1.r, c2.r, c2.r
mul r0.a, r0.a, r0.a
mul_pp r0.rgb, r0, r0.a
mul_pp r0.rgb, r0, c0
mov r0.a, t5.a
mov oC0, r0
