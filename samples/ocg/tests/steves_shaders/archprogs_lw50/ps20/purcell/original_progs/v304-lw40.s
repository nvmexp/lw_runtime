vs_1_1
dcl_position0 v0
dcl_blendweight0 v2
dcl_texcoord0 v3
dcl_texcoord1 v4
dcl_texcoord2 v5
dcl_texcoord3 v6
mov a0.x, v2.y
dp4 r0.x, c[a0.x+0], v0
dp4 r0.y, c[a0.x+1], v0
dp4 r0.z, c[a0.x+2], v0
mov a0.x, v2.w
dp4 r2.x, c[a0.x+0], v0
dp4 r2.y, c[a0.x+1], v0
dp4 r2.z, c[a0.x+2], v0
mul r0.xyz, r0.xyz, v2.x
mad r0.xyz, r2.xyz, v2.z, r0.xyz
mov r0.w, v0.w
mov a0.x, v5.y
dp4 r3.x, c[a0.x+0], v3
dp4 r3.y, c[a0.x+1], v3
dp4 r3.z, c[a0.x+2], v3
mov a0.x, v5.w
dp4 r4.x, c[a0.x+0], v3
dp4 r4.y, c[a0.x+1], v3
dp4 r4.z, c[a0.x+2], v3
mul r3.xyz, r3.xyz, v5.x
mad r3.xyz, r4.xyz, v5.z, r3.xyz
mov a0.x, v6.y
dp4 r5.x, c[a0.x+0], v4
dp4 r5.y, c[a0.x+1], v4
dp4 r5.z, c[a0.x+2], v4
mov a0.x, v6.w
dp4 r6.x, c[a0.x+0], v4
dp4 r6.y, c[a0.x+1], v4
dp4 r6.z, c[a0.x+2], v4
mul r5.xyz, r5.xyz, v6.x
mad r5.xyz, r6.xyz, v6.z, r5.xyz
add r3.xyz, r3.xyz, -r0.xyz
add r5.xyz, r5.xyz, -r0.xyz
mul r1.xyz, r3.yzx, r5.zxy
mad r1.xyz, -r5.yzx, r3.zxy, r1.xyz
add r2.xyz, c[4].xyz, -r0.xyz
dp3 r2.w, r2.xyz, r1.xyz
slt r2.w, r2.w, c[4].w
add r3.w, v0.w, -r2.w
add r1.xyz, r0.xyz, -c[4].xyz
mov r1.w, c[4].w
mul r2, r1, r2.w
mad r2, r0, r3.w, r2
m4x4 oPos, r2, c[0]
