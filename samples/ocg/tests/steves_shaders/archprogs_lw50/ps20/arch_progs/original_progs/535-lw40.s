;; Id: 535   pixel count: 3292 lw40 ppc: 2.0
ps_2_0
def c2, 16.000000, 0.000000, 0.000000, 0.000000
dcl t0.rg
dcl t2.rg
dcl t3.rgb
dcl t6.rgb
dcl v0.rgb
dcl_2d s0
dcl_lwbe s1
dcl_2d s4
mov r0.rgb, t6
dp3 r1.r, r0, t3
dp3 r0.r, t6, t6
add r0.a, r1.r, r1.r
mul r0.rgb, r0.r, t3
mad r0.rgb, r0.a, t6, -r0
texld r2, r0, s1
texld r1, t2, s4
texld r0, t0, s0
mul r2.rgb, r2.a, r2
mul r1.rgb, r1, r2
mul r1.rgb, r1, c0
mul r2.rgb, r1, c2.r
mul r1.rgb, v0, c1
mad r0.rgb, r0, r1, r2
mul r0.a, r0.a, c1.a
mov oC0, r0
