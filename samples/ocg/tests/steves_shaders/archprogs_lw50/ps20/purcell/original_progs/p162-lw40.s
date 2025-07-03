ps_1_4
texld r2, t2
texcrd r4.rgb, t4
texld r3, t5
dp3_sat r3.g, r2_bx2, r3
mov r3.b, c0.r
dp3_sat r3.r, r2_bx2, r4
phase
texcrd r2.rgb, t1
texld r0, t0
texld r1, t0
texld r5, r3
mul r3, r3.r, r0
mad r3, r5, r1, r3
dp3_sat r0, r2, r2
mul r0, 1-r0.r, r3
mul_x2 r0, r0, v0
