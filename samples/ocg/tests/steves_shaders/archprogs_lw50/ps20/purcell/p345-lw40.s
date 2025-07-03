ps_1_4
texld r0, t0
texcrd r1.rgb, t1
texcrd r2.rgb, t2
texcrd r3.rgb, t3
texld r4, t4
dp3 r5.r, r1, r0_bx2
dp3 r5.g, r2, r0_bx2
dp3 r5.b, r3, r0_bx2
dp3_x2 r3.rgb, r5, r4_bx2
mul r3.rgb, r5, r3
dp3 r2.rgb, r5, r5
mad r2.rgb, -r4_bx2, r2, r3
mov r5, r0.a
phase
texld r3, r2
dp3_sat r1, v0_bx2, r0_bx2
mul r0.rgb, r3, c0
mul r1.rgb, r0, r0
+mul r0.a, 1-r1.a, 1-r1.a
lrp r0.rgb, c1, r1, r0
+mul r0.a, r0.a, r0.a
dp3 r1.rgb, r0, c3
+mul r0.a, r0.a, 1-r1.a
lrp r0.rgb, c2, r0, r1
mad r0.a, r0.a, c6.a, c4.a
mul r0.rgb, r0, r0.a
+mov r0.a, r5.r
