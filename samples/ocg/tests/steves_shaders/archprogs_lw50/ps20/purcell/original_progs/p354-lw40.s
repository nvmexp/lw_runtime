ps_1_1
tex t0
tex t1
mul r0.rgb, t1, t0
+mov r0.a, 1-t1
mul_x2 r0.rgb, c0, r0
