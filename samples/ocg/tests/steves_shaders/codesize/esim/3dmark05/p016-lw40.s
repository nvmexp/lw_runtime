ps_3_0

def c5, 1.000000, 0.000977, 0.001953, -0.000977
def c6, 1.000000, 0.000000, 0.500000, 16.000000
def c7, 2.000000, -1.000000, 0.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord1_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3.rgb
dcl_lwbe s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
add r0.rgb, -v3, c5.gbaa
texld r0, r0, s0
add r1.rgb, -v3, c5.baga
texld r1, r1, s0
mov r0.g, r1.r
dp3 r0.a, v3, v3
add r0.rg, r0, -r0.a
add_sat_pp r1.a, -r0.a, c5.r
cmp r0.rg, r0, c6.r, c6.g
dp2add_pp r2.a, r0, c6.b, c6.g
dp3_pp r0.a, v1, v1
rsq_pp r1.b, r0.a
nrm_pp r5.rgb, v2
mov r0.a, c1.r
mul r0, r0.a, c0
mad_pp r1.rgb, v1, r1.b, r5
mul r0, r1.a, r0
nrm_pp r2.rgb, r1
texld r1, v0, s3
mad_pp r4.rgb, c7.r, r1.agba, c7.g
mul_pp r0, r2.a, r0
dp3_pp r3.r, r4, r2
texld_pp r1, v0, s2
mul_pp r3.g, r1.a, c4.r
mul_pp r2.rgb, r1, c3
texld_pp r1, r3, s4
mov_pp r2.a, r3.g
mul_pp r1, r0, r1.r
dp3_pp r3.a, r4, r5
mul_pp r1, r2, r1
mul_sat_pp r2.a, r3.a, c6.a
mov_sat_pp r3.a, r3.a
mul_pp r2, r1, r2.a
mul_pp r0, r0, r3.a
texld_pp r1, v0, s1
mul_pp r1, r1, c2
mad oC0, r0, r1, r2

