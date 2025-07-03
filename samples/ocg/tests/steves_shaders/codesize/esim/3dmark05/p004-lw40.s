ps_3_0

def c5, 0.000000, 1.000000, 2.000000, -1.000000
def c6, 16.000000, 0.000000, 0.000000, 0.000000
def c7, 0.000000, 0.000000, 0.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord4_pp v1.rgb
dcl_texcoord6 v2.rgb
dcl_texcoord7 v3.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
cmp r0.a, -v2.b, c5.r, c5.g
if_ne r0.a, -r0.a
dp3_pp r0.a, v1, v1
rsq_pp r0.b, r0.a
nrm_pp r5.rgb, v2
dp3 r0.a, v3, v3
mad_pp r0.rgb, v1, r0.b, r5
add_sat_pp r1.a, -r0.a, c5.g
nrm_pp r1.rgb, r0
texld r0, v0.rgrr, s2
mad_pp r4.rgb, c5.b, r0.agba, c5.a
mov r0.a, c1.r
mul r0, r0.a, c0
dp3_pp r2.r, r4, r1
mul_pp r0, r1.a, r0
texld_pp r3, v0.rgrr, s1
mul_pp r2.a, r3.a, c4.r
mov_pp r2.gb, c5.r
texldl_pp r1, r2.rabg, s3
mul_pp r1, r0, r1.r
mul_pp r2.rgb, r3, c3
dp3_pp r3.a, r4, r5
mul_pp r1, r2, r1
mul_sat_pp r2.a, r3.a, c6.r
mul_pp r2, r1, r2.a
mov_sat_pp r3.a, r3.a
texld_pp r1, v0.rgrr, s0
mul_pp r1, r1, c2
mul_pp r0, r0, r3.a
mad oC0, r0, r1, r2
else
mov oC0, c5.r
endif

