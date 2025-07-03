ps_3_0

def c5, 2.000000, -1.000000, 16.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord7 v2.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dp3_pp r0.a, v1, v1
rsq_pp r0.a, r0.a
nrm_pp r1.rgb, v2
mad_pp r0.rgb, v1, r0.a, r1
nrm_pp r2.rgb, r0
texld r0, v0, s2
mad_pp r0.rgb, c5.r, r0.agba, c5.g
dp3_pp r3.r, r0, r2
dp3_pp r3.a, r0, r1
texld_pp r0, v0, s1
mul_pp r3.g, r0.a, c4.r
mul_pp r2.rgb, r0, c3
texld_pp r1, r3, s3
mov r0.a, c1.r
mul_pp r0, r0.a, c0
mov_pp r2.a, r3.g
mul_pp r1, r1.r, r0
mul_pp r1, r2, r1
mul_sat_pp r2.a, r3.a, c5.b
mov_sat_pp r3.a, r3.a
mul_pp r2, r1, r2.a
texld_pp r1, v0, s0
mul_pp r1, r1, c2
mul_pp r0, r0, r3.a
mad oC0, r0, r1, r2

