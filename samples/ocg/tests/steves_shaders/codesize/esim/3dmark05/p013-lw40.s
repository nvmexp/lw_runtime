ps_3_0

def c6, 2.000000, -1.000000, 16.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord2_pp v2.rgb
dcl_texcoord6 v3.rgb
dcl_texcoord7 v4
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
dcl_2d s6
dp3_pp r0.a, v2, v2
rsq_pp r0.a, r0.a
nrm_pp r1.rgb, v3
mad_pp r0.rgb, v2, r0.a, r1
nrm_pp r2.rgb, r0
texld r0, v0, s3
mad_pp r0.rgb, c6.r, r0.agba, c6.g
dp3_pp r4.r, r0, r2
dp3_pp r4.a, r0, r1
texld_pp r0, v0, s2
mul_pp r4.g, r0.a, c4.r
mul_pp r2.rgb, r0, c3
texld_pp r0, r4, s6
texldp_pp r3, v4, s0
mov r0.a, c1.r
mul r1, r0.a, c0
mov_pp r2.a, r4.g
mul_pp r1, r3.r, r1
mul_pp r0, r0.r, r1
mov_sat_pp r3.a, r4.a
mul_pp r0, r2, r0
mul_pp r1, r1, r3.a
texld_pp r2, v0, s1
mul_pp r3, r2, c2
mul_sat_pp r5.a, r4.a, c6.b
mul_pp r2, r1, r3
texld_pp r1, v1, s4
mul_pp r4, r2, r1
texld_pp r2, v0, s5
mul_pp r2, r2, c5.r
mul_pp r0, r0, r5.a
mad r2, r2, r3, r4
mad oC0, r0, r1, r2

