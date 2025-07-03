ps_2_0
def c0, -0.500000, 0.000000, 0.000000, 0.030000
dcl v0.r
dcl_pp t0.rg
dcl_pp t1.rgb
dcl_pp t2.rgb
dcl_pp t3.rgb
dcl_pp t4.rgb
dcl_2d s0
dcl_2d s1
texld_pp r0, t0, s1
texld_pp r7, t0, s0
add_pp r0.rgb, r0, c0.r
mov_pp r2.a, r0.a
dp3_pp r9.r, t2, r0
dp3_pp r9.g, t3, r0
dp3_pp r9.b, t4, r0
dp3 r9.a, r9, r9
rsq r9.a, r9.a
mul_pp r0.rgb, r9, r9.a
mad_pp r11.rgb, r0, c0.a, t1
mov_pp r11.a, c0.g
mov_pp oC0, r11
mov_pp r0.a, v0.r
mov_pp oC1, r0
mov_pp r2.rgb, r7
mov_pp oC2, r2
