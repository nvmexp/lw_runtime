ps_2_0
def c3, 2.000000, -1.000000, 0.000000, 1.000000
dcl t0.rg
dcl t1.rgb
dcl t2.rgb
dcl t3.rgb
dcl t4.rgb
dcl_lwbe s0
dcl_2d s1
texld r0, t0, s1
mad r0.rgb, c3.r, r0, c3.g
mul r1.rgb, r0.g, t3
mad r1.rgb, t2, r0.r, r1
mad r2.rgb, t4, r0.b, r1
dp3 r0.r, r2, r2
rcp r0.a, r0.r
dp3 r0.r, r2, t1
mul r0.a, r0.a, r0.r
add r0.a, r0.a, r0.a
mad r0.rgb, r0.a, r2, -t1
texld r0, r0, s0
dp3 r3.r, t1, t1
rsq r0.a, r3.r
mul r1.rgb, r0.a, t1
dp3 r1.r, r1, r2
cmp r0.a, r1.r, r1.r, c3.b
add r2.a, -r0.a, c3.a
rsq r4.a, r3.r
mul r0.a, r4.a, r3.r
mul r1.a, r2.a, r2.a
mul r1.a, r1.a, r1.a
mov r3.a, -c1.r
mul r2.a, r2.a, r1.a
add r1.a, r3.a, c2.r
add r0.a, r0.a, r3.a
rcp r1.a, r1.a
mad r0.rgb, r0, r2.a, c0
mul_sat r0.a, r0.a, r1.a
mov oC0, r0
