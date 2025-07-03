ps_2_0
def c2, -0.004000, 1.000000, 0.250000, 0.000000
def c3, 1.000000, -1.000000, 0.454545, 256.000000
def c4, 0.500000, -0.500000, 0.000000, -2.000000
def c5, 2.000000, -1.000000, 0.000000, 0.000000
dcl t1
dcl t2.rgb
dcl t3.rgb
dcl t4.rgb
dcl_2d s0
dcl_2d s1
dcl_lwbe s4
dcl_2d s7
rcp r0.a, t1.a
mad r2.rg, r0.a, t1, c3
mul r9.rg, r2, c4
add r4.rg, r9, c0
texld r11, r4, s7
texld r6, r4, s1
texld r1, r4, s0
texld r8, t2, s4
dp3 r11.a, t4, t4
rsq r11.a, r11.a
mul r3.rgb, r11.a, t4
dp3 r11.a, t3, t3
rsq r11.a, r11.a
mul r5.rgb, r11.a, t3
mad r7.rgb, c5.r, r11, c5.g
dp3 r7.a, r7, r5
mul r5.a, r7.a, c4.a
cmp r1.a, r7.a, r7.a, c2.a
mad r9.rgb, r7, r5.a, r5
dp3 r4.a, r9, -r3
cmp r3.a, r4.a, r4.a, c2.a
log r10.a, r3.a
mul r6.a, r6.a, c3.a
mul r6.a, r10.a, r6.a
exp r6.a, r6.a
mul r11.rgb, r6, r6.a
mad r7.rgb, r1, r1.a, r11
mul r5.rgb, r7, c1
dp3 r5.a, t2, t2
rsq r9.a, r5.a
mul r1.a, r9.a, r5.a
add r2.a, r1.a, c2.r
add r8, r8, -r2.a
cmp_pp r4, r8, c2.a, c2.g
dp4 r3.a, r4, c2.b
add r6.a, -r3.a, c2.g
mul r5.rgb, r5, r6.a
add r5.a, -r5.a, c2.g
cmp r5.a, r5.a, r5.a, c2.a
mul r5.a, r5.a, r5.a
mul r11.rgb, r5, r5.a
log r0.r, r11.r
log r0.g, r11.g
log r0.b, r11.b
mul r7.rgb, r0, c3.b
exp r1.r, r7.r
exp r1.g, r7.g
exp r1.b, r7.b
mov r1.a, c2.g
mov oC0, r1
