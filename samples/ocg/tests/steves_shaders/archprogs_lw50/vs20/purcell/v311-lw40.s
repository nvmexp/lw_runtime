vs_1_1
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_blendweight0 v5
mov a0.x, v5.y
dp4 r0.x, c[a0.x+0], v0
dp4 r0.y, c[a0.x+1], v0
dp4 r0.z, c[a0.x+2], v0
dp3 r1.x, c[a0.x+0], v1
dp3 r1.y, c[a0.x+1], v1
dp3 r1.z, c[a0.x+2], v1
dp3 r2.x, c[a0.x+0], v2
dp3 r2.y, c[a0.x+1], v2
dp3 r2.z, c[a0.x+2], v2
dp3 r3.x, c[a0.x+0], v3
dp3 r3.y, c[a0.x+1], v3
dp3 r3.z, c[a0.x+2], v3
mov a0.x, v5.w
dp4 r4.x, c[a0.x+0], v0
dp4 r4.y, c[a0.x+1], v0
dp4 r4.z, c[a0.x+2], v0
dp3 r5.x, c[a0.x+0], v1
dp3 r5.y, c[a0.x+1], v1
dp3 r5.z, c[a0.x+2], v1
dp3 r6.x, c[a0.x+0], v2
dp3 r6.y, c[a0.x+1], v2
dp3 r6.z, c[a0.x+2], v2
dp3 r7.x, c[a0.x+0], v3
dp3 r7.y, c[a0.x+1], v3
dp3 r7.z, c[a0.x+2], v3
mul r0.xyz, r0.xyz, v5.x
mad r0.xyz, r4.xyz, v5.z, r0.xyz
mov r0.w, v0.w
m4x4 oPos, r0, c[0]
mul r1.xyz, r1.xyz, v5.x
mad r1.xyz, r5.xyz, v5.z, r1.xyz
dp3 r1.w, r1.xyz, r1.xyz
rsq r1.w, r1.w
mul r1.xyz, r1.xyz, r1.w
mul r2.xyz, r2.xyz, v5.x
mad r2.xyz, r6.xyz, v5.z, r2.xyz
dp3 r2.w, r2.xyz, r2.xyz
rsq r2.w, r2.w
mul r2.xyz, r2.xyz, r2.w
mul r3.xyz, r3.xyz, v5.x
mad r3.xyz, r7.xyz, v5.z, r3.xyz
dp3 r3.w, r3.xyz, r3.xyz
rsq r3.w, r3.w
mul r3.xyz, r3.xyz, r3.w
mov oT0.xy, v4.xy
add r4.xyz, c[7].xyz, -r0.xyz
dp3 r4.w, r4.xyz, r4.xyz
rsq r4.w, r4.w
mul r4.xyz, r4.xyz, r4.w
dp3 r5.x, r4.xyz, r1.xyz
dp3 r5.y, r4.xyz, r2.xyz
dp3 r5.z, r4.xyz, r3.xyz
add r6.xyz, c[13].xyz, -r0.xyz
dp3 r6.w, r6.xyz, r6.xyz
rsq r6.w, r6.w
mul r6.xyz, r6.xyz, r6.w
dp3 r7.x, r6.xyz, r1.xyz
dp3 r7.y, r6.xyz, r2.xyz
dp3 r7.z, r6.xyz, r3.xyz
add r8.xyz, r5.xyz, r7.xyz
dp3 r8.w, r8.xyz, r8.xyz
rsq r8.w, r8.w
mul r8.xyz, r8.xyz, r8.w
mov r1.x, r5.z
mov r1.y, r8.z
mov r1.w, c[11].w
lit r2, r1
dp3 r3.xy, r3.xyz, r4.xyz
mov r3.w, c[9].w
lit r3, r3
mul r4.xyz, r3.z, c[12].xyz
add r4.xyz, r4.xyz, c[11].xyz
mul oD0.xyz, r4.xyz, r2.z
mov oT0.xy, v4
mov oT1.xy, v4
mov oT2.xyz, r5
mov oT3.xyz, r8
