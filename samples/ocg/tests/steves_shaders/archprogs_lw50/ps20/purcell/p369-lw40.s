ps_2_0
def c0, 0.000000, 0.000000, 0.000000, 1.000000
def c1, 0.501961, 0.501961, 0.501961, 1.000000
def c2, 0.219520, 0.219520, 0.219520, 0.058594
def c3, 0.000000, 0.000000, 0.000000, 0.500000
def c4, 2.000000, -1.000000, 0.000000, 0.000000
dcl t2.rg
dcl_2d s0
texld r0, t2, s0
mad r2.rgb, c4.r, r0, c4.g
dp3 r2.a, r2, r2
rsq r2.a, r2.a
mul r9.rgb, r2, r2.a
mad r11.rgb, r9, c3.a, c3.a
mov r11.a, c0.a
mov oC3, r11
mov r6, c0
mov oC0, r6
mov r1, c1
mov oC1, r1
mov r8, c2
mov oC2, r8
