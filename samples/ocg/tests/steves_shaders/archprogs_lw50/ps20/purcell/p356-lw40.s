ps_2_0
def c0, 0.000000, 0.000000, 0.000000, 0.000000
dcl v0.r
dcl_pp t0.rg
dcl_pp t1.rgb
dcl_pp t2.rgb
dcl_2d s0
texld_pp r0, t0, s0
mov_pp r7.rgb, t1
mov_pp r7.a, c0.r
mov_pp oC0, r7
dp3 r0.a, t2, t2
rsq r0.a, r0.a
mul_pp r2.rgb, r0.a, t2
mov_pp r2.a, v0.r
mov_pp oC1, r2
mov_pp r0.a, c0.r
mov_pp oC2, r0
