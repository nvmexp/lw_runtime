ps_2_0
dcl t0.rg
dcl_2d s0
add r10.rg, t0, c16.abgr
add r9.rg, t0, c16
add r8.rg, t0, c17
add r7.rg, t0, c17.abgr
add r6.rg, t0, c18
add r5.rg, t0, c18.abgr
add r4.rg, t0, c19
add r3.rg, t0, c19.abgr
add r2.rg, t0, c20
add r1.rg, t0, c20.abgr
add r0.rg, t0, c21
texld_pp r10, r10, s0
texld_pp r9, r9, s0
texld_pp r8, r8, s0
texld_pp r7, r7, s0
texld_pp r6, r6, s0
texld_pp r5, r5, s0
texld_pp r4, r4, s0
texld_pp r3, r3, s0
texld_pp r2, r2, s0
texld_pp r1, r1, s0
texld_pp r0, r0, s0
mul_pp r10, r10, c1
mad_pp r9, r9, c0, r10
mad_pp r8, r8, c2, r9
mad_pp r7, r7, c3, r8
mad_pp r6, r6, c4, r7
mad_pp r5, r5, c5, r6
mad_pp r4, r4, c6, r5
mad_pp r3, r3, c7, r4
mad_pp r2, r2, c8, r3
mad_pp r1, r1, c9, r2
mad_pp r5, r0, c10, r1
add r4.rg, t0, c21.abgr
add r3.rg, t0, c22
add r2.rg, t0, c22.abgr
add r1.rg, t0, c23
add r0.rg, t0, c23.abgr
texld_pp r4, r4, s0
texld_pp r3, r3, s0
texld_pp r2, r2, s0
texld_pp r1, r1, s0
texld_pp r0, r0, s0
mad_pp r4, r4, c11, r5
mad_pp r3, r3, c12, r4
mad_pp r2, r2, c13, r3
mad_pp r1, r1, c14, r2
mad_pp r0, r0, c15, r1
mov_pp oC0, r0
