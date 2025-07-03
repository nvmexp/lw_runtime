ps_2_0
dcl_pp t0.rg
dcl_2d s0
texld_pp r0, t0, s0
mul_pp r0.a, r0.a, c1.a
mul_pp r0.rgb, r0, c1
mov_pp oC0, r0
