ps_3_0

def c5, 1.000000, 2.000000, -1.000000, 16.000000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dp3 r0.a, v3, v3
add_sat_pp r1.a, -r0.a, c5.r
dp3_pp r0.a, v1, v1
rsq_pp r1.b, r0.a
nrm_pp r2.rgb, v2
mov r0.a, c1.r
mul r0, r0.a, c0
mad_pp r1.rgb, v1, r1.b, r2
mul_pp r0, r1.a, r0
nrm_pp r3.rgb, r1
texld r1, v0, s2
mad_pp r1.rgb, c5.g, r1.agba, c5.b
dp3_pp r3.r, r1, r3
dp3_pp r3.a, r1, r2
texld_pp r1, v0, s1
mul_pp r3.g, r1.a, c4.r
mul_pp r2.rgb, r1, c3
texld_pp r1, r3, s3
mov_pp r2.a, r3.g
mul_pp r1, r0, r1.r
mul_pp r1, r2, r1
mul_sat_pp r2.a, r3.a, c5.a
mov_sat_pp r3.a, r3.a
mul_pp r2, r1, r2.a
texld_pp r1, v0, s0
mul_pp r1, r1, c2
mul_pp r0, r0, r3.a
mad oC0, r0, r1, r2

