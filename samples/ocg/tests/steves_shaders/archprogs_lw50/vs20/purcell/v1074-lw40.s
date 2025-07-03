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
mad r0, c[a0.x+42], v1.x, r0
dp4 r4.x, v0, r0
mul r1, v1.y, c[a0.y+43]
mad r2, c[a0.x+43], v1.x, r1
mul r1, v1.y, c[a0.y+44]
dp4 r4.y, v0, r2
mad r1, c[a0.x+44], v1.x, r1
dp4 r4.z, v0, r1
mov r4.w, c[0].y
dp4 oPos.x, r4, c[8]
dp4 oPos.y, r4, c[9]
dp4 oPos.w, r4, c[11]
dp4 r3.y, r4, c[10]
add r5.xyz, -r4, c[29]
mad oFog, -r3.y, c[16].w, c[16].x
dp3 r3.z, r5, r5
mov oPos.z, r3.y
rsq r3.w, r3.z
dst r3.xy, r3.z, r3.w
add oT3.xyz, -r4, c[2]
dp4 r0.w, c[31], r3
mul r6.xyz, r5, r3.w
rcp r0.w, r0.w
dp3 r8.y, v6, r2
dp3 r7.y, v3, r2
dp3 r8.z, v6, r1
dp3 r7.z, v3, r1
dp3 r7.x, v3, r0
dp3 r8.x, v6, r0
mul r1.xyz, r0.w, c[27]
mul r0.xyz, r8.yzxw, r7.zxyw
mul r2.xy, r8.yzzw, c[3]
mad r0.xyz, r7.yzxw, r8.zxyw, -r0
add r5.x, r2.y, r2.x
mul r0.xyz, r0, v6.w
mul r3.xy, r7.yzzw, c[3]
mul r2.xy, r0.yzzw, c[3]
add r5.z, r3.y, r3.x
add r5.y, r2.y, r2.x
dp3 r2.y, r5, r6
dp3 r4.x, c[1].yzzw, r8
dp3 r4.z, c[1].yzzw, r7
dp3 r4.y, c[1].yzzw, r0
dp3 r2.x, r4, r6
dp3 r3.x, c[1].w, r8
mov oT4.x, r8.x
mov oT5.x, r8.y
mov oT6.x, r8.z
dp3 r3.z, c[1].w, r7
mov oT4.z, r7.x
mov oT5.z, r7.y
mov oT6.z, r7.z
dp3 r3.y, c[1].w, r0
dp3 r2.z, r3, r6
slt r6.xyz, r4, c[0].x
mul r4.xyz, r4, r4
mova a0.xyz, r6
max r2.xyz, r2, c[0].x
mul r6.xyz, r4.y, c[a0.y+23]
mad r6.xyz, r4.x, c[a0.x+21], r6
slt r7.xyz, r5, c[0].x
mul r5.xyz, r5, r5
mova a0.xyw, r7.xyz
mad r4.xyz, r4.z, c[a0.z+25], r6
mul r6.xyz, r5.y, c[a0.y+23]
mad r6.xyz, r5.x, c[a0.x+21], r6
slt r7.xyz, r3, c[0].x
mul r3.xyz, r3, r3
mova a0.xyz, r7
mad oD0.xyz, r1, r2.x, r4
mul r4.xyz, r3.y, c[a0.y+23]
mad r5.xyz, r5.z, c[a0.w+25], r6
mad r4.xyz, r3.x, c[a0.x+21], r4
mad oD1.xyz, r1, r2.y, r5
mad r3.xyz, r3.z, c[a0.z+25], r4
mad oT7.xyz, r1, r2.z, r3
mul r1.xy, v4, c[90]
add r0.w, r1.y, r1.x
mul r1.xy, v4, c[91]
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
