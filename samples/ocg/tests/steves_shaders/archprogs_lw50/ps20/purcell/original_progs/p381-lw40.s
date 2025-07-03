ps_2_0
def c0, 0.816497, 0.000000, 0.577350, 0.000000
def c2, -0.408248, 0.707107, 0.577350, 0.000000
def c3, -0.408248, -0.707107, 0.577350, 0.000000
def c5, 2.000000, -1.000000, 0.000000, 0.000000
dcl_pp t0.rg
dcl_pp t1.rg
dcl v0.rgb
dcl v1.rgb
dcl_pp t7.rgb
dcl_2d s0
dcl_2d s3
texld_pp r1, t1, s3
texld_pp r0, t0, s0
mad_pp r2.rgb, c5.r, r1, c5.g
dp3_sat r1.r, r2, c2
mul r1.rgb, r1.r, v1
dp3_sat r3.r, r2, c0
dp3_sat r2.r, r2, c3
mad r1.rgb, r3.r, v0, r1
mad_pp r1.rgb, r2.r, t7, r1
mul_pp r1.rgb, r1, c1
mul_pp r0.rgb, r0, r1
mul_pp r0.rgb, r0, c4
mul_pp r0.a, r0.a, c1.a
add_pp r0.rgb, r0, r0
mov_pp oC0, r0
