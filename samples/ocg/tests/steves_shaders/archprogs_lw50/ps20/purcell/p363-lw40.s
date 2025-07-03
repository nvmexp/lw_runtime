ps_2_0
def c10, 1.000000, 0.000000, 0.125000, -0.070000
dcl_pp t0
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
dcl_2d s3
texldp_pp r0, t0, s0
add_pp r2.rgb, r0, -c9
texld_pp r9, r2, s2
dp3 r0.a, r2, r2
rsq r10.a, r0.a
mul_pp r0.a, r10.a, r0.a
add_pp r2.a, -r9.r, r0.a
mad_pp r11.rgb, r2.a, c0, r2
mad_pp r1.rgb, r2.a, c1, r2
mad_pp r3.rgb, r2.a, c2, r2
mad_pp r5.rgb, r2.a, c3, r2
mad_pp r9.rgb, r2.a, c4, r2
mad_pp r6.rgb, r2.a, c5, r2
mad_pp r7.rgb, r2.a, c6, r2
mad_pp r4.rgb, r2.a, c7, r2
texld_pp r11, r11, s2
texld_pp r1, r1, s2
texld_pp r3, r3, s2
texld_pp r5, r5, s2
texld_pp r9, r9, s2
texld_pp r6, r6, s2
texld_pp r8, r7, s2
texld_pp r10, r4, s2
add_pp r10.a, r0.a, c10.a
add_pp r7.r, -r11.r, r10.a
add_pp r7.g, r10.a, -r1.r
texldp_pp r4, t0, s1
rcp_pp r4.a, r0.a
mul_pp r2.rgb, r2, r4.a
dp3 r4.a, r0, r0
rsq r4.a, r4.a
mad_pp r0.rgb, r0, -r4.a, -r2
dp3_pp r2.r, -r2, r4
dp3 r4.a, r0, r0
rsq r4.a, r4.a
mul_pp r0.rgb, r0, r4.a
dp3_pp r2.g, r0, r4
texld_pp r11, r2, s3
add_pp r7.b, r10.a, -r3.r
add_pp r7.a, r10.a, -r5.r
cmp_pp r3, -r7, c10.r, c10.g
dp4_pp r8.a, r3, c10.b
add_pp r5.r, r10.a, -r9.r
add_pp r5.g, r10.a, -r6.r
add_pp r5.b, r10.a, -r8.r
add_pp r5.a, r10.a, -r10.r
cmp_pp r7, -r5, c10.r, c10.g
dp4_pp r1.a, r7, c10.b
add_pp r3.a, r8.a, r1.a
mul_pp r0.a, r0.a, c9.a
mul_pp r11, r11, c8
mul_pp r8, r3.a, r11
add_pp r4.a, -r0.a, c10.r
mul_pp r2, r8, r4.a
mov_pp oC0, r2
