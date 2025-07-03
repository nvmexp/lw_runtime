ps_2_0
def c10, 1.000000, 0.000000, -65500.000000, 65500.000000
def c11, 0.500001, -1.000000, 0.000000, 0.000000
dcl t0.rg
dcl_2d s0
texld r0, t0, s0
add r1.a, -r0.a, c10.r
cmp_pp r1.a, r1.a, c10.r, c10.g
add_pp r2.a, r0.a, c10.b
add r1.rg, t0, -c4.abgr
cmp_pp r2.a, r2.a, c10.r, c10.g
rcp r2.r, c4.r
rcp r2.g, c4.g
add_pp r1.a, r1.a, r2.a
mul r1.rg, r1, r2
cmp r0.a, -r1.a, r0.a, c10.a
mul r2.rg, r1, r0.a
mul r1.rgb, r2.g, c1
mad r1.rgb, c0, r2.r, r1
mad r1.rgb, c2, r0.a, r1
add r2.rgb, r1, c3
add r1.a, r2.b, -c5.b
rcp r2.a, r1.a
mov r1.a, c7.r
add r1.a, r1.a, -c5.b
mul r2.a, r2.a, r1.a
lrp r1.rgb, r2.a, r2, c5
add r2.a, -r2.b, c7.r
cmp r2.rgb, r2.a, r2, r1
cmp r1.rgb, r1.a, c5, r1
add r1.rgb, -r2, r1
dp3 r2.r, r1, r1
pow r1.a, r2.r, c11.r
mul r2.a, r1.a, c8.r
pow r1.a, c6.r, r2.a
add r2.a, r1.a, c11.g
mul r1.rgb, r2.a, c9
mad r0.rgb, r0, r1.a, r1
mov oC0, r0
