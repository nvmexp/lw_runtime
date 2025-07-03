ps_2_0
def c2, -1.000000, 0.000000, 0.000000, 0.000000
dcl_pp t0.rg
dcl_pp t1.rg
dcl_pp t2.rg
dcl_pp t3.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld_pp r0, t0, s2
texld_pp r7, t1, s2
texld_pp r2, t2, s2
texld_pp r9, t3, s2
texld_pp r4, t0, s0
texld_pp r11, t1, s0
texld_pp r6, t2, s0
texld_pp r1, t3, s0
texld_pp r8, t0, s1
texld_pp r3, t1, s1
texld_pp r10, t2, s1
texld_pp r5, t3, s1
add_pp r0, r0, r7
add_pp r7, r2, r0
add_pp r2, r9, r7
mul_pp r9, r2, c0
add_pp r4.a, r4.a, r11.a
add_pp r11.a, r6.a, r4.a
add_pp r6.a, r1.a, r11.a
mad_pp r0, r6.a, c1, r9
add_pp r8, r8, r3
add_pp r3, r10, r8
add_pp r10, r5, r3
mul_pp r5, r10, r0
add_pp r7, r5.a, r5
add_pp r2, r7, c2.r
mov_pp oC0, r2
