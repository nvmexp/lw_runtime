//;; Id: 518   pixel count: 2949120 lw40 ppc: 2.66666666667
ps_2_0
def c0, -0.062745, 0.250000, 0.000000, 0.000000
dcl t0.rg
dcl t1.rg
dcl t2.rg
dcl t3.rg
dcl_2d s0
texld r3, t0, s0
texld r4, t1, s0
texld r2, t2, s0
texld r1, t3, s0
add_sat r0.r, r3.a, c0.r
add_sat r0.g, r4.a, c0.r
add r3.rgb, r3, r4
add_sat r0.b, r2.a, c0.r
add r2.rgb, r2, r3
add_sat r0.a, r1.a, c0.r
add r1.rgb, r1, r2
dp4 r0.a, r0, c0.g
mul r0.rgb, r1, c0.g
mov oC0, r0
