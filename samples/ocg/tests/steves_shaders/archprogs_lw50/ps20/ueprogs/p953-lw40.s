ps_2_0
def c1, -0.500000, 0.000000, 0.000000, 0.000000
def c2, 0.700000, 0.600000, 0.500000, 0.000000
dcl t1.rg
dcl t5
dcl_2d s0
texld r1, t1, s0
add r0, r1.a, c1.r
mul r1.rgb, r1, c0
texkill r0
mul r0.rgb, r1, c2
mov r0.a, t5.a
mov oC0, r0
