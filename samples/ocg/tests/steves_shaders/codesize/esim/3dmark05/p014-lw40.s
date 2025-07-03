ps_3_0

def c6, 2.000000, -1.000000, 16.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_2d s5
dp3_pp r0.a, v1, v1
rsq_pp r0.a, r0.a
nrm_pp r2.rgb, v2
mad_pp r0.rgb, v1, r0.a, r2
nrm_pp r1.rgb, r0
texld r0, v0.rgrr, s3
mad_pp r0.rgb, c6.r, r0.agba, c6.g
dp3_pp r1.r, r0, r1
dp3_pp r4.a, r0, r2
mov r1.ba, c6.a
texld_pp r0, v0.rgrr, s2
mul_pp r1.g, r0.a, c4.r
mul_pp r3.rgb, r0, c3
texldl_pp r0, r1, s5
mov_pp r3.a, r1.g
texldp_pp r2, v3, s0
mov r0.a, c1.r
mul r1, r0.a, c0
mul_pp r1, r2.r, r1
mov_sat_pp r2.a, r4.a
mul_pp r0, r0.r, r1
mul_pp r1, r1, r2.a
texld r2, v0.rgrr, s1
mul_pp r2, r2, c2
mul_pp r0, r3, r0
mul_pp r3, r1, r2
texld r1, v0.rgrr, s4
mul_pp r1, r1, c5.r
mul_sat_pp r4.a, r4.a, c6.b
mad r1, r1, r2, r3
mad oC0, r0, r4.a, r1

