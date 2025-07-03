vs_2_0
def c[243], 3.000000, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_texcoord0 v1
dcl_tangent0 v2
dcl_binormal0 v3
dcl_normal0 v4
dcl_blendindices0 v5
dcl_blendweight0 v6
mul r2, v5, c[243].x
mova a0, r2
m4x3 r7.xyz, v0, c[a0.x+0]
mul r4.xyz, r7, v6.x
m4x3 r9.xyz, v0, c[a0.y+0]
mad r11.xyz, r9, v6.y, r4
m4x3 r6.xyz, v0, c[a0.z+0]
mad r1.xyz, r6, v6.z, r11
m4x3 r3.xyz, v0, c[a0.w+0]
mad r5.xyz, r3, v6.w, r1
mul r2, r5.x, c[229]
mad r4, r5.y, c[230], r2
mad r0, r5.z, c[231], r4
add oT2, r0, c[232]
m3x3 r11.xyz, v2, c[a0.x+0]
mul r6.xyz, r11, v6.x
m3x3 r8.xyz, v2, c[a0.y+0]
mad r3.xyz, r8, v6.y, r6
m3x3 r10.xyz, v2, c[a0.z+0]
mad r7.xyz, r10, v6.z, r3
m3x3 r4.xyz, v2, c[a0.w+0]
mad r0.xyz, r4, v6.w, r7
mul r11, r5.x, c[225]
mul r6, r5.y, c[226]
mul r5, r5.z, c[227]
add r8, r11, r6
add r1, r5, r8
add r3, r1, c[228]
mad r7.xyz, r3, -c[240].w, c[240]
mul r4.xyz, r7.x, c[237]
mad r6.xyz, r7.y, c[238], r4
mad r5.xyz, r7.z, c[239], r6
m3x3 r8.xyz, v3, c[a0.x+0]
mul r1.xyz, r8, v6.x
m3x3 r10.xyz, v3, c[a0.y+0]
mad r7.xyz, r10, v6.y, r1
m3x3 r4.xyz, v3, c[a0.z+0]
mad r11.xyz, r4, v6.z, r7
m3x3 r6.xyz, v3, c[a0.w+0]
mad r1.xyz, r6, v6.w, r11
m3x3 r10.xyz, v4, c[a0.w+0]
mul r2.xyz, r10, v6.w
m3x3 r7.xyz, v4, c[a0.z+0]
m3x3 r9.xyz, v4, c[a0.x+0]
mul r11.xyz, r9, v6.x
m3x3 r6.xyz, v4, c[a0.y+0]
mad r10.xyz, r6, v6.y, r11
mad r7.xyz, r7, v6.z, r10
add r2.xyz, r2, r7
m3x3 oT3.xyz, r5, r0
add r4.xyz, -r3, c[241]
mul r9.xyz, r4.x, c[237]
mad r8.xyz, r4.y, c[238], r9
mad r10.xyz, r4.z, c[239], r8
m3x3 oT4.xyz, r10, r0
mov r10.xyz, c[237]
mul r7.xyz, c[242].x, r10
mov r10.xyz, c[238]
mad r4.xyz, c[242].y, r10, r7
mov r10.xyz, c[239]
mad r11.xyz, c[242].z, r10, r4
m3x3 oT5.xyz, r11, r0
mul r0, r3.w, c[236]
mul r1, r3.x, c[233]
mul r2, r3.y, c[234]
mul r3, r3.z, c[235]
add r6, r1, r2
add r8, r3, r6
add r10, r0, r8
mov oPos, r10
mov oT1, r10
mov oT0.xy, v1
