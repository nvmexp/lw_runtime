ps_2_0
def c2, 16.000000, 0.000000, 0.000000, 0.000000
dcl t3.rgb
dcl t6.rgb
dcl_lwbe s1
mov r0.rgb, t6
dp3 r1.r, r0, t3
dp3 r0.r, t6, t6
add r0.a, r1.r, r1.r
mul r0.rgb, r0.r, t3
mad r0.rgb, r0.a, t6, -r0
texld r0, r0, s1
mul r0.rgb, r0.a, r0
mul r0.rgb, r0, c0
mul r0.rgb, r0, c2.r
mov r0.a, c1.a
mov oC0, r0
