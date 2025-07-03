vs_2_0
def c[229], 3.000000, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_blendweight0 v2
mul r0, v1, c[229].x
mova a0, r0
m4x3 r7.xyz, v0, c[a0.x+0]
mul r2.xyz, r7, v2.x
m4x3 r9.xyz, v0, c[a0.y+0]
mad r11.xyz, r9, v2.y, r2
m4x3 r6.xyz, v0, c[a0.z+0]
mad r8.xyz, r6, v2.z, r11
m4x3 r3.xyz, v0, c[a0.w+0]
mad r5.xyz, r3, v2.w, r8
mul r0, r5.x, c[225]
mad r9, r5.y, c[226], r0
mad r4, r5.z, c[227], r9
add r11, r4, c[228]
mov oPos, r11
mov oT0.xyz, r11.xyw
