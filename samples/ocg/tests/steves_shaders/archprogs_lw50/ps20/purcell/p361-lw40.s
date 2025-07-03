ps_2_0
dcl_pp t0.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld_pp r0, t0, s0
texld_pp r7, t0, s1
dp3 r0.a, r0, r0
rsq r0.a, r0.a
mad_pp r11.rgb, r0, -r0.a, -c0
dp3 r11.a, r11, r11
rsq r11.a, r11.a
mul_pp r6.rgb, r11, r11.a
dp3_pp r1.g, r6, r7
dp3_pp r1.r, -c0, r7
texld_pp r8, r1, s2
mul_pp r3, r8, c1
mov_pp oC0, r3
