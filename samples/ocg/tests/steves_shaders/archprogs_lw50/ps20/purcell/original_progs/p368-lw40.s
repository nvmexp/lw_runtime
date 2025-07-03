ps_2_0
def c0, -0.000975, 0.000975, 0.000000, 0.000000
def c1, 0.000975, -0.000975, 0.000000, 0.000000
dcl t0.rg
dcl_2d s0
add r0.rg, t0, c0.r
add r7.rg, t0, c1
add r2.rg, t0, c0
add r9.rg, t0, c0.g
texld r4, r0, s0
texld r11, r7, s0
texld r6, r2, s0
texld r1, r9, s0
mov r4.g, r11.r
mov r4.b, r6.r
mov r4.a, r1.r
mov oC0, r4
