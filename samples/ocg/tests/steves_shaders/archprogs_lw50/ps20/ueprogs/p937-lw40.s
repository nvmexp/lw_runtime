ps_2_0
def c2, 0.000000, 0.000000, 0.000000, 0.000000
dcl t0.rg
dcl_2d s0
dcl_2d s1
mov r0.rg, c2.r
texld_pp r1, r0, s1
texld_pp r0, t0, s0
mul r0, r0, c0
mad r2, r0, -r1.r, c1
mul r2, r2, c1.a
mad_pp r0, r0, r1.r, r2
mov_pp oC0, r0
