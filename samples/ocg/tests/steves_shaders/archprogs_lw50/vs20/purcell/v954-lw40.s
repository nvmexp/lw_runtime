vs_2_0
def c[234], 3.000000, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_blendindices0 v2
dcl_blendweight0 v3
dcl_texcoord0 v4
mul r0, v2, c[234].x
mova a0, r0
m3x3 r7.xyz, v1, c[a0.x+0]
mul r2.xyz, r7, v3.x
m3x3 r9.xyz, v1, c[a0.y+0]
mad r11.xyz, r9, v3.y, r2
m3x3 r6.xyz, v1, c[a0.z+0]
mad r8.xyz, r6, v3.z, r11
m3x3 r3.xyz, v1, c[a0.w+0]
mad oT0.xyz, r3, v3.w, r8
m4x3 r5.xyz, v0, c[a0.w+0]
m4x3 r7.xyz, v0, c[a0.z+0]
m4x3 r9.xyz, v0, c[a0.x+0]
mul r4.xyz, r9, v3.x
m4x3 r11.xyz, v0, c[a0.y+0]
mad r1.xyz, r11, v3.y, r4
mad r8.xyz, r7, v3.z, r1
mad r3.xyz, r5, v3.w, r8
mul r10, r3.x, c[225]
mad r0, r3.y, c[226], r10
mad oT1.xyz, r3, -c[233].w, c[233]
mad r9, r3.z, c[227], r0
add r4, r9, c[228]
mul r11, r4.x, c[229]
mad r1, r4.y, c[230], r11
mad r10, r4.z, c[231], r1
mad oPos, r4.w, c[232], r10
mov oT2.xy, v4
