vs_2_0
def c[1], 765.005859, 0.816497, -0.408248, 0.577350
def c[3], 0.707107, -0.707107, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
mul r0.xy, v2.zyzw, c[1].x
mova a0.xy, r0
mul r0, v1.y, c[a0.y+42]
mad r3, c[a0.x+42], v1.x, r0
dp4 r0.x, v0, r3
mul r1, v1.y, c[a0.y+43]
mad r2, c[a0.x+43], v1.x, r1
mul r1, v1.y, c[a0.y+44]
dp4 r0.y, v0, r2
mad r1, c[a0.x+44], v1.x, r1
dp4 r0.z, v0, r1
mov r0.w, c[0].y
dp3 r4.x, v3, r3
dp3 r5.x, v6, r3
dp3 r4.y, v3, r2
dp3 r5.y, v6, r2
dp3 r4.z, v3, r1
dp3 r5.z, v6, r1
dp4 oPos.x, r0, c[8]
mul r1.xyz, r4.zxyw, r5.yzxw
dp4 oPos.y, r0, c[9]
mad r1.xyz, r4.yzxw, r5.zxyw, -r1
dp4 oPos.w, r0, c[11]
mul r1.xyz, r1, v6.w
dp4 r6.y, r0, c[10]
dp3 r3.y, c[1].yzzw, r1
dp3 r3.x, c[1].yzzw, r5
dp3 r3.z, c[1].yzzw, r4
add oT3.xyz, -r0, c[2]
slt r0.xyz, r3, c[0].x
mova a0.xyz, r0
mul r0.xyz, r3, r3
mad oFog, -r6.y, c[16].w, c[16].x
mul r2.xyz, r0.y, c[a0.y+23]
mov oPos.z, r6.y
mad r2.xyz, r0.x, c[a0.x+21], r2
dp3 r0.x, r3, -c[28]
mad r3.xyz, r0.z, c[a0.z+25], r2
dp3 r2.x, c[1].w, r5
mov oT4.x, r5.x
dp3 r2.z, c[1].w, r4
dp3 r2.y, c[1].w, r1
mov oT4.z, r4.x
dp3 r0.z, r2, -c[28]
mul r6.xy, r5.yzzw, c[3]
mov oT5.x, r5.y
mov oT6.x, r5.z
add r4.x, r6.y, r6.x
mul r6.xy, r4.yzzw, c[3]
mov oT5.z, r4.y
mov oT6.z, r4.z
mul r5.xy, r1.yzzw, c[3]
add r4.z, r6.y, r6.x
add r4.y, r5.y, r5.x
dp3 r0.y, r4, -c[28]
slt r5.xyz, r4, c[0].x
mul r4.xyz, r4, r4
mova a0.xyw, r5.xyz
max r0.xyz, r0, c[0].x
mul r5.xyz, r4.y, c[a0.y+23]
mad r5.xyz, r4.x, c[a0.x+21], r5
slt r6.xyz, r2, c[0].x
mul r2.xyz, r2, r2
mova a0.xyz, r6
mad oD0.xyz, r0.x, c[27], r3
mul r3.xyz, r2.y, c[a0.y+23]
mad r4.xyz, r4.z, c[a0.w+25], r5
mad r3.xyz, r2.x, c[a0.x+21], r3
mad oD1.xyz, r0.y, c[27], r4
mad r2.xyz, r2.z, c[a0.z+25], r3
mad oT7.xyz, r0.z, c[27], r2
mul r0.xy, v4, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v4, c[91]
mov oT0.x, r0.w
add r0.z, r0.y, r0.x
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oT4.y, r1.x
mov oT5.y, r1.y
mov oT6.y, r1.z
mov oD0.w, c[0].x
mov oD1.w, c[0].x
