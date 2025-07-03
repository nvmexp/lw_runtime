ps_2_0
dcl_pp t0.rg
dcl_pp t1
dcl_pp t2
dcl_pp t3
dcl_pp t4
dcl_pp t5
dcl_pp t6
dcl_pp t7
dcl_2d s0
mov_pp r0.rg, t1.abgr
mov_pp r7.rg, t2.abgr
mov_pp r2.rg, t3.abgr
mov_pp r9.rg, t4.abgr
texld_pp r4, t0, s0
texld_pp r11, t1, s0
texld_pp r6, r0, s0
texld_pp r1, t2, s0
texld_pp r8, r7, s0
texld_pp r3, t3, s0
texld_pp r10, r2, s0
texld_pp r5, t4, s0
texld_pp r0, r9, s0
texld_pp r7, t5, s0
mul_pp r2, r4, c1.a
mad_pp r4, r11, c0.r, r2
mad_pp r6, r6, c0.r, r4
mad_pp r2, r1, c0.g, r6
mad_pp r9, r8, c0.g, r2
mad_pp r4, r3, c0.b, r9
mov_pp r11.rg, t5.abgr
mov_pp r6.rg, t6.abgr
mov_pp r1.rg, t7.abgr
texld_pp r2, r11, s0
texld_pp r8, t6, s0
texld_pp r9, r6, s0
texld_pp r3, t7, s0
texld_pp r11, r1, s0
mad_pp r4, r10, c0.b, r4
mad_pp r6, r5, c0.a, r4
mad_pp r1, r0, c0.a, r6
mad_pp r10, r7, c1.r, r1
mad_pp r4, r2, c1.r, r10
mad_pp r5, r8, c1.g, r4
mad_pp r6, r9, c1.g, r5
mad_pp r0, r3, c1.b, r6
mad_pp r1, r11, c1.b, r0
mov_pp oC0, r1
