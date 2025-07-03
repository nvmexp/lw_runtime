;; Id: 567   pixel count: 823162 lw40 ppc: 2.0
ps_2_0
def c1, 2.000000, -1.000000, 0.000000, 1.000000
dcl t0.rg
dcl t1.rgb
dcl t2.rgb
dcl t3.rgb
dcl t4.rgb
dcl_lwbe s0
dcl_2d s1
dcl_lwbe s6
texld r1, t1, s6
texld r0, t0, s1
mad r1.rgb, c1.r, r1, c1.g
mad r2.rgb, c1.r, r0, c1.g
dp3 r0.r, r2, t2
dp3 r0.g, r2, t3
dp3 r0.b, r2, t4
dp3 r3.r, r0, r0
dp3 r2.r, r0, r1
mul r1.rgb, r1, r3.r
add r0.a, r2.r, r2.r
max r1.a, r2.r, c1.b
mad r0.rgb, r0.a, r0, -r1
texld r0, r0, s0
add r0.a, -r1.a, c1.a
mul r1.a, r0.a, r0.a
mul r1.a, r1.a, r1.a
mul r0.a, r0.a, r1.a
mad r0.rgb, r0, r0.a, c0
mov r0.a, c1.a
mov oC0, r0
