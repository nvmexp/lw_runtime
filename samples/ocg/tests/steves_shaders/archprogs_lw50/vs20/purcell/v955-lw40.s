vs_2_0
dcl_position0 v0
dcl_texcoord0 v1
dcl_tangent0 v2
dcl_binormal0 v3
dcl_normal0 v4
mul r0, v0.x, c[4]
mad r2, v0.y, c[5], r0
mad r4, v0.z, c[6], r2
mad oT2, v0.w, c[7], r4
mul r6, v0.x, c[0]
mad r8, v0.y, c[1], r6
mad r10, v0.z, c[2], r8
mad r0, v0.w, c[3], r10
mad r9.xyz, r0, -c[15].w, c[15]
mul r4.xyz, r9.x, c[12]
mad r1.xyz, r9.y, c[13], r4
mad r8.xyz, r9.z, c[14], r1
m3x3 oT3.xyz, r8, v2
add r10.xyz, -r0, c[16]
mul r5.xyz, r10.x, c[12]
mad r9.xyz, r10.y, c[13], r5
mad r4.xyz, r10.z, c[14], r9
m3x3 oT4.xyz, r4, v2
mov r4.xyz, c[12]
mul r11.xyz, c[17].x, r4
mov r4.xyz, c[13]
mad r1.xyz, c[17].y, r4, r11
mov r4.xyz, c[14]
mad r3.xyz, c[17].z, r4, r1
m3x3 oT5.xyz, r3, v2
mul r10, r0.w, c[11]
mul r5, r0.x, c[8]
mul r7, r0.y, c[9]
mul r0, r0.z, c[10]
add r2, r5, r7
add r9, r0, r2
add r4, r10, r9
mov oPos, r4
mov oT1, r4
mov oT0.xy, v1
