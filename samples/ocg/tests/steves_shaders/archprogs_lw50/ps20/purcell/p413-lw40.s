ps_2_0
def c1, 2.000000, -1.000000, 0.000000, 1.000000
dcl t0.rg
dcl t1.rgb
dcl t2.rgb
dcl t3.rgb
dcl t4.rgb
dcl_lwbe s0
dcl_2d s1
texld r0, t0, s1
mad r0.rgb, c1.r, r0, c1.g
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
nrm r1.rgb, t1
dp3 r1.r, r1, r2
cmp r0.a, r1.r, r1.r, c1.b
add r1.a, -r0.a, c1.a
mul r0.a, r1.a, r1.a
mul r0.a, r0.a, r0.a
mul r0.a, r1.a, r0.a
mad r0.rgb, r0, r0.a, c0
mov r0.a, c1.a
mov oC0, r0
