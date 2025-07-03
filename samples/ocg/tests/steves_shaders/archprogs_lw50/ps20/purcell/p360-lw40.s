ps_2_0
def c10, 1.000000, 0.250000, 0.333333, 0.000000
def c11, 0.010000, 0.000000, 0.000000, 0.000000
dcl_pp t0.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld_pp r0, t0, s0
mov_pp r7.rgb, r0
mul r0.rg, r0.b, c11.r
mov_pp r7.a, c10.r
dp4_pp r0.a, c9, r7
rcp_pp r0.a, r0.a
dp4_pp r2.r, c6, r7
dp4_pp r2.g, c7, r7
dp4_pp r2.b, c8, r7
mul_pp r9.rgb, r0.a, r2
add_pp r4.rg, r9, c0
add_pp r11.rg, r9, c0.abgr
add_pp r6.rg, r9, c1
add_pp r1.rg, r9, c1.abgr
add_pp r8.rg, r9, c2
add_pp r3.rg, r9, c2.abgr
texld_pp r10, r4, s1
texld_pp r5, r11, s1
texld_pp r7, r6, s1
texld_pp r2, r1, s1
texld_pp r4, r8, s1
texld_pp r11, r3, s1
add_pp r6.r, r9.b, -r10.r
add_pp r6.g, r9.b, -r5.r
add_pp r6.b, r9.b, -r7.r
add_pp r6.a, r9.b, -r2.r
cmp_pp r8, -r6, c10.r, c10.a
dp4_pp r3.r, r8, c10.g
add_pp r10.r, r9.b, -r4.r
mov_pp r9.a, -r11.r
add_pp r5.rg, r9, c3
add_pp r7.rg, r9, c3.abgr
add_pp r2.rg, r9, c4
add_pp r4.rg, r9, c4.abgr
add_pp r11.rg, r9, c5
add_pp r9.rg, r9, c5.abgr
texld_pp r6, r5, s1
texld_pp r1, r7, s1
texld_pp r8, r2, s1
texld_pp r5, r4, s1
texld_pp r7, r11, s1
texld_pp r2, r9, s1
texld_pp r0, r0, s2
add_pp r10.g, r9.b, r9.a
add_pp r10.b, r9.b, -r6.r
add_pp r10.a, r9.b, -r1.r
cmp_pp r11, -r10, c10.r, c10.a
dp4_pp r3.g, r11, c10.g
add_pp r6.r, r9.b, -r8.r
add_pp r6.g, r9.b, -r5.r
add_pp r6.b, r9.b, -r7.r
add_pp r6.a, r9.b, -r2.r
cmp_pp r1, -r6, c10.r, c10.a
dp4_pp r3.b, r1, c10.g
dp3_pp r8.a, r3, c10.b
mad_pp r7, r0.r, r8.a, r0.a
mov_pp oC0, r7
