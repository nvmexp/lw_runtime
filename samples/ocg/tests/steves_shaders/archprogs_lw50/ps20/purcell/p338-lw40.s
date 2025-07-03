ps_1_1
tex t0
tex t1
tex t2
tex t3
dp3_sat r1, t0_bx2, c0
mul r0, t1, r1
dp3_sat r1, t0_bx2, c1
mad r0, r1, t2, r0
dp3_sat r1, t0_bx2, c2
mad r0, r1, t3, r0
mul r0, r0, c3
