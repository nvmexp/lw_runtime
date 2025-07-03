ps_3_0

def c8, 1.000000, 0.000000, 2.000000, -1.000000
def c9, 16.000000, 0.000000, 0.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dp3 r0.r, v2, c0
dp3 r0.g, v2, c1
dp3 r0.b, v2, c2
dp3 r0.a, r0, r0
add_sat_pp r1.a, -r0.a, c8.r
mov r0.a, c4.r
mul r0, r0.a, c3
mul r0, r1.a, r0
texldp r1, v3, s0
dp3_pp r2.a, v1, v1
rsq_pp r2.a, r2.a
nrm_pp r5.rgb, v2
mad_pp r3.rgb, v1, r2.a, r5
mul_pp r0, r0, r1
nrm_pp r2.rgb, r3
texld r1, v0, s3
mad_pp r4.rgb, c8.b, r1.agba, c8.a
cmp_pp r0, -v3.a, c8.g, r0
dp3_pp r3.r, r4, r2
texld_pp r1, v0, s2
mul_pp r3.g, r1.a, c7.r
mul_pp r2.rgb, r1, c6
texld_pp r1, r3, s4
mov_pp r2.a, r3.g
mul_pp r1, r0, r1.r
dp3_pp r3.a, r4, r5
mul_pp r1, r2, r1
mul_sat_pp r2.a, r3.a, c9.r
mov_sat_pp r3.a, r3.a
mul_pp r2, r1, r2.a
mul_pp r0, r0, r3.a
texld_pp r1, v0, s1
mul_pp r1, r1, c5
mad oC0, r0, r1, r2

