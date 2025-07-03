ps_3_0

def c5, 2.000000, -1.000000, 16.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord2_pp v2.rgb
dcl_texcoord3_pp v3.rgb
dcl_texcoord4_pp v4.rgb
dcl_texcoord5_pp v5.rgb
dcl_texcoord6 v6.rgb
dcl_texcoord7 v7
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_lwbe s5
dcl_2d s6
texldp_pp r1, v7, s0
dp3_pp r0.a, v5, v5
rsq_pp r1.a, r0.a
nrm_pp r5.rgb, v6
mov r0.a, c1.r
mul r0, r0.a, c0
mad_pp r2.rgb, v5, r1.a, r5
mul_pp r3, r1.r, r0
nrm_pp r1.rgb, r2
texld r0, v0.rgrr, s3
mad_pp r4.rgb, c5.r, r0.agba, c5.g
mov r2.ba, c5.a
dp3_pp r2.r, r4, r1
texld_pp r0, v0.rgrr, s2
mul_pp r2.g, r0.a, c4.r
mul_pp r1.rgb, r0, c3
texldl_pp r0, r2, s6
mov_pp r1.a, r2.g
mul_pp r0, r3, r0.r
dp3_pp r2.a, r4, r5
mul_pp r0, r1, r0
mul_sat_pp r5.a, r2.a, c5.b
mov_sat_pp r4.a, r2.a
dp3_pp r1.r, r4, v2
dp3_pp r1.g, r4, v3
dp3_pp r1.b, r4, v4
texld_pp r2, r1, s5
texld_pp r1, v1.rgrr, s4
mul_pp r4, r3, r4.a
texld_pp r3, v0.rgrr, s1
mul_pp r3, r3, c2
mul_pp r4, r4, r3
mul_pp r2, r2, r1
mul_pp r4, r1, r4
mul_pp r0, r0, r5.a
mad r2, r2, r3, r4
mad oC0, r0, r1, r2

