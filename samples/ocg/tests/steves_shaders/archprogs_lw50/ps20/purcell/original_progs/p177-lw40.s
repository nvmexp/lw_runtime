ps_1_1
tex t0
tex t1
dp3 r0.rgb, t0_bx2, t1_bx2
+mov r0.a, t1_bx2
mad_sat r0, r0, c0, c1
mul r0.rgb, r0, v0
+add_x4_sat r0.a, r0, r0
mul r0.rgb, r0, r0.a
