ps_2_0
dcl t0.rg
dcl_2d s0
dcl_2d s1
texld r1, t0, s1
texld_pp r0, t0, s0
mad_pp r0, r0, c0, r1
mov_pp oC0, r0
