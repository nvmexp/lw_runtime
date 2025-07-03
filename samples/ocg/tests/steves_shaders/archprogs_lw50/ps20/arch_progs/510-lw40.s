//;; Id: 510   pixel count: 7137050 lw40 ppc: 2.0
ps_2_0
def c2, 1.000000, 16.000000, 0.062500, 0.000000
def c3, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl t3.rgb
dcl t6.rgb
dcl v0.rgb
dcl_2d s0
dcl_lwbe s1
mov r0.rgb, t6
dp3 r1.r, r0, t3
dp3 r0.r, t6, t6
add r0.a, r1.r, r1.r
mul r0.rgb, r0.r, t3
mad r0.rgb, r0.a, t6, -r0
texld r1, r0, s1
texld r0, t0, s0
mul r1.rgb, r1.a, r1
add r0.a, -r0.a, c2.r
mul r1.rgb, r1, r0.a
mul r1.rgb, r1, c0
mul r2.rgb, r1, c2.g
mul r1.rgb, v0, c1
mad r0.rgb, r0, r1, r2
dp3 r1.r, r0, c3
mul r0.a, r1.r, c2.b
mov oC0, r0
