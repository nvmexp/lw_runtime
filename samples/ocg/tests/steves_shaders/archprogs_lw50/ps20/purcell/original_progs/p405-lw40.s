ps_2_0
def c1, 0.816497, 0.000000, 0.577350, 0.000000
def c7, -0.408248, 0.707107, 0.577350, 0.000000
def c8, -0.408248, -0.707107, 0.577350, 0.000000
def c9, 2.000000, -1.000000, 1.000000, 0.333333
dcl_pp t0.rg
dcl_pp t1.rg
dcl_pp t2
dcl_pp t3
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7.rgb
dcl v0
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
dcl_2d s4
texld_pp r5, t1, s4
mad_pp r5.rgb, c9.r, r5, c9.g
mul r0.rgb, r5.g, t6
mad r0.rgb, t5, r5.r, r0
mad r7.rgb, t7, r5.b, r0
dp3 r0.r, r7, r7
rcp r0.a, r0.r
dp3 r0.r, r7, t4
mul r0.a, r0.a, r0.r
add r0.a, r0.a, r0.a
mad r0.rgb, r0.a, r7, -t4
mov_pp r2.r, t2.b
mov_pp r2.g, t2.a
mov_pp r1.r, t3.b
mov_pp r1.g, t3.a
texld_pp r4, r0, s2
texld_pp r3, r2, s1
texld_pp r2, t2, s1
texld_pp r1, r1, s1
texld_pp r0, t0, s0
mul_pp r6.rgb, r5.a, r4
nrm r4.rgb, t4
mul_pp r6.rgb, r6, c0
dp3 r7.r, r7, r4
mad_pp r4.rgb, r6, r6, -r6
add r2.a, -r7.r, c9.b
mad_pp r6.rgb, c2, r4, r6
mul r1.a, r2.a, r2.a
dp3_pp r7.r, r6, c9.a
mul r1.a, r1.a, r1.a
lrp_pp r4.rgb, c3, r6, r7.r
mul r1.a, r2.a, r1.a
mov_pp r6.r, c5.a
mad r1.a, r1.a, r6.r, c4.a
dp3_sat r6.r, r5, c7
mul r3.rgb, r3, r6.r
dp3_sat r6.r, r5, c1
dp3_sat r5.r, r5, c8
mad r2.rgb, r6.r, r2, r3
mad_pp r2.rgb, r5.r, r1, r2
mul_pp r0.rgb, r0, v0
mul_pp r1.rgb, r4, r1.a
mul_pp r0.rgb, r2, r0
mul_pp r0.a, r0.a, v0.a
mad_pp r0.rgb, r0, c6.r, r1
mov_pp oC0, r0
