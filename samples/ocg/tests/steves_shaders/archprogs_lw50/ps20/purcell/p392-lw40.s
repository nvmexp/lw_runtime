ps_2_0
dcl_pp t0.rg
dcl v0.rgb
dcl_2d s0
texld_pp r0, t0, s0
mul_pp r1.rgb, v0, c1
mul_pp r0.rgb, r0, r1
mul_pp r0.rgb, r0, c4
mul_pp r0.a, r0.a, c1.a
add_pp r0.rgb, r0, r0
mov_pp oC0, r0
