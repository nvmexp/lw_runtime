ps_1_4
texld r2, t2
texcrd r4.rgb, t4
dp3_sat r3.r, r2_bx2, r4
phase
texcrd r2.rgb, t1
texld r0, t0
mul r3, r3.r, r0
dp3_sat r0, r2, r2
mul r0, 1-r0.r, r3
mul_x2 r0, r0, v0
