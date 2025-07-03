ps_2_0
def c2, -1.000000, 0.500000, 0.000000, -2.000000
dcl_pp t0.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
texld_pp r0, t0, s2
texld_pp r7, t0, s0
texld_pp r2, t0, s1
texld_pp r9, t0, s3
mul_pp r4, r0, c0
mad_pp r6, r7.a, c1, r4
mul_pp r1, r2, r6
mul_pp r8.a, r1.a, c2.a
add_pp r3, r1.a, r1
mul_pp r8.rgb, r3, c2.r
add_pp r10, r9, r8
mad_pp r0, r10, c2.g, r3
mov_pp oC0, r0
