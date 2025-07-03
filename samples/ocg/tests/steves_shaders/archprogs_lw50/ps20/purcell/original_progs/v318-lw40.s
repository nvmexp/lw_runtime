vs_1_1
dcl_position0 v0
dcl_texcoord0 v1
dcl_blendweight0 v2
dcl_texcoord2 v3
mov a0.x, v2.y
dp4 r0.x, c[a0.x+0], v0
dp4 r0.y, c[a0.x+1], v0
dp4 r0.z, c[a0.x+2], v0
mul r2.xyz, c[a0.x+3], v2.x
mul r3.xyz, c[a0.x+4], v2.x
mov a0.x, v2.w
dp4 r1.x, c[a0.x+0], v0
dp4 r1.y, c[a0.x+1], v0
dp4 r1.z, c[a0.x+2], v0
mad r2.xyz, c[a0.x+3], v2.z, r2.xyz
mad r3.xyz, c[a0.x+4], v2.z, r3.xyz
mul r0.xyz, r0.xyz, v2.x
mad r0.xyz, r1.xyz, v2.z, r0.xyz
mov r0.w, v0.w
m4x4 oPos, r0, c[0]
mov oT0, v1
add r5.xyz, r2.xyz, -v0.xyz
dp3 r5.w, r5.xyz, r5.xyz
rsq r5.w, r5.w
mul r5.xyz, r5.xyz, r5.w
mov oT4.xyz, r5
add r7.xyz, r3.xyz, -r0.xyz
dp3 r7.w, r7.xyz, r7.xyz
rsq r7.w, r7.w
mul r7.xyz, r7.xyz, r7.w
add r8.xyz, r5.xyz, r7.xyz
dp3 r8.w, r8.xyz, r8.xyz
rsq r8.w, r8.w
mul oT5.xyz, r8.xyz, r8.w
mov oD0, c[8]
dp4 oT1.x, r0, c[13]
dp4 oT1.y, r0, c[14]
dp4 oT1.z, r0, c[15]
mov oT1.w, r0.w
mov oT2, v3
mov oT3, v0
