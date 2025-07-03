ps_3_0

def c5, 2.000000, -1.000000, 16.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord2_pp v1.rgb
dcl_texcoord3_pp v2.rgb
dcl_texcoord1_pp v3.rgb
dcl_texcoord4_pp v4.rgb
dcl_texcoord6 v5.rgb
dcl_texcoord7 v6
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_lwbe s4
dcl_2d s5
texld r0, v0.rgrr, s3
mad_pp r6.rgb, c5.r, r0.agba, c5.g
dp3_pp r0.r, r6, v1
dp3_pp r0.g, r6, v2
dp3_pp r0.b, r6, v3
texld_pp r1, r0, s4
texld_pp r0, v0.rgrr, s1
mul_pp r3, r0, c2
texldp_pp r2, v6, s0
mov r0.a, c1.r
mul r0, r0.a, c0
nrm_pp r4.rgb, v5
mul_pp r5, r2.r, r0
dp3_pp r0.g, r6, r4
mov_sat_pp r0.a, r0.g
dp3_pp r0.b, v4, v4
mul_sat_pp r7.a, r0.g, c5.b
rsq_pp r2.a, r0.b
mul_pp r0, r5, r0.a
mad_pp r2.rgb, v4, r2.a, r4
mul_pp r4, r3, r0
nrm_pp r0.rgb, r2
dp3_pp r6.r, r6, r0
mov r6.ba, c5.a
texld_pp r0, v0.rgrr, s2
mul_pp r6.g, r0.a, c4.r
mul_pp r2.rgb, r0, c3
texldl_pp r0, r6, s5
mov_pp r2.a, r6.g
mul_pp r0, r5, r0.r
mad r1, r1, r3, r4
mul_pp r0, r2, r0
mad oC0, r0, r7.a, r1

