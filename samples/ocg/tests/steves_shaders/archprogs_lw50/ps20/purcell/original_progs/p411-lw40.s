ps_2_0
def c1, 1.000000, 0.333333, 0.000000, 0.000000
dcl_pp t0.rg
dcl_pp t2.rg
dcl_pp t3.rg
dcl t4.rgb
dcl t7.rgb
dcl v0
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
dcl_2d s5
dp3 r0.r, t7, t7
rcp r0.a, r0.r
mov r0.rgb, t7
dp3 r1.r, r0, t4
mul r0.a, r0.a, r1.r
add r0.a, r0.a, r0.a
mad r0.rgb, r0.a, r0, -t4
texld_pp r2, r0, s2
texld_pp r3, t3, s5
texld_pp r1, t2, s1
texld_pp r0, t0, s0
mul_pp r3.rgb, r2, r3
nrm r2.rgb, t4
mul_pp r3.rgb, r3, c0
dp3 r4.r, t7, r2
mad_pp r2.rgb, r3, r3, -r3
add r2.a, -r4.r, c1.r
mad_pp r3.rgb, c2, r2, r3
mul r1.a, r2.a, r2.a
dp3_pp r4.r, r3, c1.g
mul r1.a, r1.a, r1.a
lrp_pp r2.rgb, c3, r3, r4.r
mul r1.a, r2.a, r1.a
mov_pp r3.r, c5.a
mad r1.a, r1.a, r3.r, c4.a
mul_pp r0.rgb, r0, v0
mul_pp r2.rgb, r2, r1.a
mul_pp r0.rgb, r1, r0
mul_pp r0.a, r0.a, v0.a
mad_pp r0.rgb, r0, c6.r, r2
mov_pp oC0, r0
