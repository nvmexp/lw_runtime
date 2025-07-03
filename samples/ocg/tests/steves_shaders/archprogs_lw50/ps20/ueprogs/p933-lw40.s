ps_2_0
def c2, 0.222000, 0.444000, 0.111000, 0.000000
dcl t0.rg
dcl_2d s0
add r3.rg, t0, c0.abgr
add r2.rg, t0, c0
add r1.rg, t0, c1
add r0.rg, t0, c1.abgr
texld r3, r3, s0
texld r2, r2, s0
texld r1, r1, s0
texld r0, r0, s0
mul_pp r3, r3, c2.r
mad_pp r2, r2, c2.g, r3
mad_pp r1, r1, c2.r, r2
mad_pp r0, r0, c2.b, r1
mov_pp oC0, r0
