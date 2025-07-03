ps_2_0
def c2, -0.500000, 0.000000, 15.000000, 1.000000
def c3, 0.020000, -0.008000, 2.000000, -1.000000
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
texld r0, t1, s0
dp3 r0.r, t6, t6
rsq r4.a, r0.r
mad r0.a, r0.a, c3.r, c3.g
mul_pp r4.rgb, r4.a, t6
mad r2.rg, r4, r0.a, t1
texld r1, r2, s1
add r0, r1.a, c2.r
rcp r1.a, t7.a
mul r3.rg, r1.a, t7
mad r5.rg, r3, c0, c0.abgr
texkill r0
texld r3, r2, s0
texld_pp r2, r2, s2
texld r0, r5, s3
mad_pp r5.rgb, c3.b, r3, c3.a
nrm_pp r3.rgb, r5
dp3_pp r4.r, r3, r4
mul_pp r4.rgb, r3, r4.r
add_pp r4.rgb, r4, r4
mad_pp r5.rgb, t6, -r4.a, r4
nrm_pp r4.rgb, t4
dp3 r5.r, r5, r4
max r1.a, r5.r, c2.g
dp3 r3.r, r3, r4
pow r0.a, r1.a, c2.b
max r1.a, r3.r, c2.g
dp3_pp r3.r, t5, t5
add r2.a, -r3.r, c2.a
mul r2.rgb, r2, r0.a
max r0.a, r2.a, c2.g
mad r1.rgb, r1, r1.a, r2
mul r0.a, r0.a, r0.a
mul_pp r1.rgb, r1, r0.a
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c1
mov r0.a, c2.g
mov oC0, r0
