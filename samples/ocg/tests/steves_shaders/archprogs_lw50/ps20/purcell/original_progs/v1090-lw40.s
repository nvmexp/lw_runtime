vs_2_0
def c[1], 765.005859, 0.816497, -0.408248, 0.577350
def c[3], 0.707107, -0.707107, 0.000000, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_normal0 v2
dcl_texcoord0 v3
dcl_tangent0 v5
mul r0.w, v1.z, c[1].x
mova a0.w, r0.w
mov r0.w, c[0].y
dp4 r0.x, v0, c[a0.w+42]
dp4 r0.y, v0, c[a0.w+43]
dp4 r0.z, v0, c[a0.w+44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp3 r5.x, v2, c[a0.w+42]
dp3 r6.x, v5, c[a0.w+42]
dp3 r5.y, v2, c[a0.w+43]
dp3 r5.z, v2, c[a0.w+44]
dp3 r6.y, v5, c[a0.w+43]
dp3 r6.z, v5, c[a0.w+44]
dp4 oPos.w, r0, c[11]
mul r1.xyz, r5.zxyw, r6.yzxw
dp4 r2.y, r0, c[10]
mad r1.xyz, r5.yzxw, r6.zxyw, -r1
add oT3.xyz, -r0, c[2]
mul r0.xyz, r1, v5.w
mad oFog, -r2.y, c[16].w, c[16].x
mul r1.xy, r0.yzzw, c[3]
mov oPos.z, r2.y
add r4.y, r1.y, r1.x
mul r2.xy, r6.yzzw, c[3]
mul r1.xy, r5.yzzw, c[3]
add r4.x, r2.y, r2.x
add r4.z, r1.y, r1.x
dp3 r1.y, r4, -c[28]
dp3 r3.x, c[1].yzzw, r6
dp3 r3.z, c[1].yzzw, r5
dp3 r3.y, c[1].yzzw, r0
dp3 r1.x, r3, -c[28]
dp3 r2.x, c[1].w, r6
mov oT4.x, r6.x
mov oT5.x, r6.y
mov oT6.x, r6.z
dp3 r2.z, c[1].w, r5
mov oT4.z, r5.x
mov oT5.z, r5.y
mov oT6.z, r5.z
dp3 r2.y, c[1].w, r0
dp3 r1.z, r2, -c[28]
slt r5.xyz, r3, c[0].x
mul r3.xyz, r3, r3
mova a0.xyz, r5
max r1.xyz, r1, c[0].x
mul r5.xyz, r3.y, c[a0.y+23]
mad r5.xyz, r3.x, c[a0.x+21], r5
slt r6.xyz, r4, c[0].x
mul r4.xyz, r4, r4
mova a0.xyw, r6.xyz
mad r3.xyz, r3.z, c[a0.z+25], r5
mul r5.xyz, r4.y, c[a0.y+23]
mad r5.xyz, r4.x, c[a0.x+21], r5
slt r6.xyz, r2, c[0].x
mul r2.xyz, r2, r2
mova a0.xyz, r6
mad oD0.xyz, r1.x, c[27], r3
mul r3.xyz, r2.y, c[a0.y+23]
mad r4.xyz, r4.z, c[a0.w+25], r5
mad r3.xyz, r2.x, c[a0.x+21], r3
mad oD1.xyz, r1.y, c[27], r4
mad r2.xyz, r2.z, c[a0.z+25], r3
mad oT7.xyz, r1.z, c[27], r2
mul r1.xy, v3, c[90]
add r0.w, r1.y, r1.x
mul r1.xy, v3, c[91]
mov oT0.x, r0.w
add r1.z, r1.y, r1.x
mov oT0.y, r1.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r1.z
mov oT2.y, r1.z
mov oT4.y, r0.x
mov oT5.y, r0.y
mov oT6.y, r0.z
mov oD0.w, c[0].x
mov oD1.w, c[0].x
