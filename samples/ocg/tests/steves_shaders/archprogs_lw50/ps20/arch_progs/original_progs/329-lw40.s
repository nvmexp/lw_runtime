;; Id: 329   pixel count: 72963 lw40 ppc: 4.0
ps_1_1
tex t0
tex t1
dp3 r1, t0_bx2, v1_bx2
add r0, r1, c0
mul r1, v0, r0
mul r0.rgb, r1, t1
mul r0.a, t1.a, v0.a
