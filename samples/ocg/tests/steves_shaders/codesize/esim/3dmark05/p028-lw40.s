ps_3_0

def c5, 1.000000, 0.000000, 2.000000, -1.000000
def c6, 16.000000, 0.000000, 0.000000, 0.000000
def c7, 0.000000, 0.000000, 0.000000, 0.000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_texcoord5_pp v2.rgb
dcl_texcoord7 v3.rgb
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
cmp_pp r0.a, -v3.b, c5.r, c5.g
if_ne r0.a, -r0.a
mov oC0, c5.g
else
dp3_pp r0.a, v2, v2
rsq_pp r0.a, r0.a
nrm_pp r5.rgb, v3
mad_pp r0.rgb, v2, r0.a, r5
nrm_pp r1.rgb, r0
texld r0, v0.rgrr, s2
mad_pp r4.rgb, c5.b, r0.agba, c5.a
dp3_pp r3.r, r4, r1
texld_pp r0, v0.rgrr, s1
mul_pp r3.g, r0.a, c4.r
mul_pp r2.rgb, r0, c3
mov_pp r3.ba, c5.g
texldl_pp r1, r3, s4
mov r0.a, c1.r
mul_pp r0, r0.a, c0
mov_pp r2.a, r3.g
mul_pp r1, r1.r, r0
mul_pp r2, r2, r1
dp3_pp r1.a, r4, r5
mul_sat_pp r3.a, r1.a, c6.r
mov_sat_pp r3.b, r1.a
texld_pp r1, v0.rgrr, s0
mul_pp r1, r1, c2
mul_pp r0, r0, r3.b
mul_pp r2, r2, r3.a
mul_pp r0, r1, r0
texld_pp r1, v1.rgrr, s3
mul_pp r2, r2, r1
mad oC0, r0, r1, r2
endif

