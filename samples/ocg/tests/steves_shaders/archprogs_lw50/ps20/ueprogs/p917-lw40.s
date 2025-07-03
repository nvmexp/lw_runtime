ps_2_0
def c2, 2.000000, -1.000000, 0.000000, 15.000000
def c3, 1.000000, 0.000000, 0.000000, 0.000000
dcl t1.rg
dcl t4.rgb
dcl_pp t5.rgb
dcl t6.rgb
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
texld r2, t1, s0
texld_pp r1, t1, s2
texld_pp r0, t1, s1
dp3 r3.r, t6, t6
rsq r0.a, r3.r
mul_pp r3.rgb, r0.a, t6
mad_pp r4.rgb, c2.r, r2, c2.g
nrm_pp r2.rgb, r4
dp3_pp r3.r, r2, r3
mul_pp r3.rgb, r2, r3.r
add_pp r3.rgb, r3, r3
mad_pp r4.rgb, t6, -r0.a, r3
nrm_pp r3.rgb, t4
dp3 r4.r, r4, r3
max r1.a, r4.r, c2.b
dp3 r2.r, r2, r3
pow r0.a, r1.a, c2.a
max r1.a, r2.r, c2.b
dp3_pp r2.r, t5, t5
add r2.a, -r2.r, c3.r
mul r1.rgb, r1, r0.a
max r0.a, r2.a, c2.b
mad r0.rgb, r0, r1.a, r1
mul r0.a, r0.a, r0.a
mul_pp r1.rgb, r0, r0.a
rcp r0.a, t7.a
mul r0.rg, r0.a, t7
mad r0.rg, r0, c0, c0.abgr
texld r0, r0, s3
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c1
mov r0.a, c2.b
mov oC0, r0
