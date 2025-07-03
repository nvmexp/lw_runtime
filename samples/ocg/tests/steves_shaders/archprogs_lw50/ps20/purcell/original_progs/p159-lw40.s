ps_1_4
texld r2, t2
texcrd r4.rgb, t4
dp3_sat r3.r, r2_bx2, r4
phase
texcrd r2.rgb, t1
texld r0, t0
texld r4, t3_da.rgaa
mul r3, r3.r, r0
dp3_sat r0, r2, r2
mul r0, r4, 1-r0.r
mul r0, r0, r3
cmp r1, r2.b, c0.r, c0.g
mul r0, r0, r1
mul_x2 r0, r0, v0
