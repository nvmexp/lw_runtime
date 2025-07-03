ps_1_1
tex t0
tex t1
tex t2
mov_sat r0.a, 1-t2.a
lrp r0, r0.a, t1, t0
mul r0, r0, t2
mul_x2 r0.rgb, c0, r0
