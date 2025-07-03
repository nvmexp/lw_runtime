ps_2_0
dcl_pp t0.rg
dcl_pp t2.rg
dcl v0
dcl_2d s0
dcl_2d s1
texld_pp r1, t2, s1
texld_pp r0, t0, s0
mul_pp r0.rgb, r0, v0
mul_pp r1.rgb, r1, r0
mul_pp r1.rgb, r1, c6.r
mad_pp r0.rgb, c7, r0, -r1
mad_pp r0.rgb, r0.a, r0, r1
mov_pp r0.a, v0.a
mov_pp oC0, r0
