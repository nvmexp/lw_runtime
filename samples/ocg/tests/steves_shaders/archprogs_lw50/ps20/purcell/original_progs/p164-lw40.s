ps_1_1
tex t0
tex t1
tex t2
tex t3
dp3_sat r0.rgb, t1_bx2, t3_bx2
+mov_sat r0.a, t2_bx2
mul r0.rgb, r0, r0
+add_x4_sat r0.a, r0, r0
mul r0.rgb, r0, r0
mul r1.rgb, r0.a, t0
mul r1.rgb, v0, r1
mul r0.rgb, r0, r0
mul r0.rgb, r0, r0
mul r0.rgb, r0, r1
