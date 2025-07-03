ps_3_0

def c5, 0.000000, 0.000000, 0.000000, 0.000000
def c6, 1.000000, 2.000000, -1.000000, 16.000000
dcl_texcoord0 v0.rg
dcl_texcoord5_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3.rgb
dcl_lwbe s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
dp3 r0.a, v3, v3
add_sat_pp r1.a, -r0.a, c6.r
mov r0.a, c1.r
mul r0, r0.a, c0
mul r0, r1.a, r0
texld r1, v3, s0
dp3_pp r2.a, v1, v1
rsq_pp r2.a, r2.a
nrm_pp r4.rgb, v2
mad_pp r2.rgb, v1, r2.a, r4
nrm_pp r3.rgb, r2
texld r2, v0.rgrr, s3
mad_pp r2.rgb, c6.g, r2.agba, c6.b
mul_pp r0, r0, r1
dp3_pp r3.r, r2, r3
dp3_pp r4.a, r2, r4
mov_pp r3.ba, c5.r
texld_pp r1, v0.rgrr, s2
mul_pp r3.g, r1.a, c4.r
mul_pp r2.rgb, r1, c3
texldl_pp r1, r3, s4
mov_pp r2.a, r3.g
mul_pp r1, r0, r1.r
mul_pp r1, r2, r1
mul_sat_pp r2.a, r4.a, c6.a
mov_sat_pp r3.a, r4.a
mul_pp r2, r1, r2.a
texld_pp r1, v0.rgrr, s1
mul_pp r1, r1, c2
mul_pp r0, r0, r3.a
mad r0, r0, r1, r2
cmp oC0, -v2.b, c5.r, r0

