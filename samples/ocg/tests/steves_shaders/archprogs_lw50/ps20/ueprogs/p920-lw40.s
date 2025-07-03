ps_2_0
def c2, 0.000000, 0.000000, 2.000000, -0.500000
def c3, 15.000000, 1.000000, 0.000000, 0.000000
dcl t1.rg
dcl t4.rgb
dcl_pp t5.rgb
dcl t6.rgb
dcl t7
dcl v0.r
dcl_2d s0
dcl_2d s1
texld r1, t1, s1
add r0, r1.a, c2.a
rcp r1.a, t7.a
mul r2.rg, r1.a, t7
mad r2.rg, r2, c0, c0.abgr
texkill r0
texld r0, r2, s0
nrm_pp r2.rgb, t6
mad_pp r3.rgb, r2.b, c2, -r2
nrm_pp r2.rgb, t4
dp3 r3.r, r3, r2
max r1.a, r2.b, c2.r
max r2.a, r3.r, c2.r
pow r0.a, r2.a, c3.r
dp3_pp r2.r, t5, t5
add r2.a, -r2.r, c3.g
mul r2.rgb, r1, r0.a
max r0.a, r2.a, c2.r
mad r1.rgb, r1, r1.a, r2
mul r0.a, r0.a, r0.a
mul_pp r1.rgb, r1, r0.a
mul r0.rgb, r0, v0.r
mul r0.rgb, r1, r0
mul r0.rgb, r0, c1
mov r0.a, c2.r
mov oC0, r0
