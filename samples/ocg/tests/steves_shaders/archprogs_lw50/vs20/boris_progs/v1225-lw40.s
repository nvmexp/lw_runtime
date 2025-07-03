vs_2_0
def c[1], 765.005859, 0.816497, -0.408248, 0.577350
def c[3], 0.707107, -0.707107, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
mul r0.xyz, v2.zyxw, c[1].x
mova a0.xyz, r0
add r1.w, -v1.x, c[0].y
mul r0, v1.y, c[a0.y+42]
add r3.w, r1.w, -v1.y
mad r0, c[a0.x+42], v1.x, r0
mad r2, c[a0.z+42], r3.w, r0
dp4 r3.x, v0, r2
mul r0, v1.y, c[a0.y+43]
mad r1, c[a0.x+43], v1.x, r0
mul r0, v1.y, c[a0.y+44]
mad r1, c[a0.z+43], r3.w, r1
mad r0, c[a0.x+44], v1.x, r0
dp4 r3.y, v0, r1
mad r0, c[a0.z+44], r3.w, r0
dp4 r3.z, v0, r0
mov r3.w, c[0].y
dp4 oPos.x, r3, c[8]
dp4 oPos.y, r3, c[9]
dp4 oPos.w, r3, c[11]
dp4 r4.y, r3, c[10]
add r5.xyz, -r3, c[29]
mad oFog, -r4.y, c[16].w, c[16].x
dp3 r4.z, r5, r5
mov oPos.z, r4.y
rsq r0.w, r4.z
add oT3.xyz, -r3, c[2]
mul r7.xyz, r5, r0.w
dst r4.xy, r4.z, r0.w
dp3 r4.x, c[31], r4
dp3 r3.x, -r7, c[28]
rcp r0.w, r4.x
add r1.w, r3.x, -c[30].z
dp3 r8.x, v6, r2
dp3 r5.x, v3, r2
dp3 r8.y, v6, r1
dp3 r5.y, v3, r1
dp3 r8.z, v6, r0
dp3 r5.z, v3, r0
mul r1.w, r1.w, c[30].w
mul r0.xyz, r8.yzxw, r5.zxyw
max r2.w, r1.w, c[0].x
mad r0.xyz, r5.yzxw, r8.zxyw, -r0
dp3 r1.x, c[1].yzzw, r8
mul r0.xyz, r0, v6.w
dp3 r1.z, c[1].yzzw, r5
dp3 r1.y, c[1].yzzw, r0
pow r1.w, r2.w, c[30].x
slt r2.xyz, r1, c[0].x
mova a0.xyz, r2
mul r3.xyz, r1, r1
min r1.w, r1.w, c[0].y
mul r4.xyz, r3.y, c[a0.y+23]
mul r2.xyz, r1.w, c[27]
mad r4.xyz, r3.x, c[a0.x+21], r4
mul r2.xyz, r0.w, r2
mad r6.xyz, r3.z, c[a0.z+25], r4
dp3 r4.x, c[1].w, r8
mov oT4.x, r8.x
dp3 r4.z, c[1].w, r5
dp3 r4.y, c[1].w, r0
mov oT4.z, r5.x
dp3 r3.z, r4, r7
mul r3.xy, r8.yzzw, c[3]
mov oT5.x, r8.y
mov oT6.x, r8.z
add r5.x, r3.y, r3.x
mul r8.xy, r5.yzzw, c[3]
mov oT5.z, r5.y
mov oT6.z, r5.z
mul r3.xy, r0.yzzw, c[3]
add r5.z, r8.y, r8.x
add r5.y, r3.y, r3.x
dp3 r3.y, r5, r7
dp3 r3.x, r1, r7
dp3 r1.x, r1, -c[33]
max r3.xyz, r3, c[0].x
mad r6.xyz, r2, r3.x, r6
dp3 r1.y, r5, -c[33]
dp3 r1.z, r4, -c[33]
slt r7.xyz, r5, c[0].x
mul r5.xyz, r5, r5
mova a0.xyw, r7.xyz
max r1.xyz, r1, c[0].x
mul r7.xyz, r5.y, c[a0.y+23]
mad r7.xyz, r5.x, c[a0.x+21], r7
slt r8.xyz, r4, c[0].x
mul r4.xyz, r4, r4
mova a0.xyz, r8
mad oD0.xyz, r1.x, c[32], r6
mul r6.xyz, r4.y, c[a0.y+23]
mad r5.xyz, r5.z, c[a0.w+25], r7
mad r6.xyz, r4.x, c[a0.x+21], r6
mad r5.xyz, r2, r3.y, r5
mad r4.xyz, r4.z, c[a0.z+25], r6
mad oD1.xyz, r1.y, c[32], r5
mad r2.xyz, r2, r3.z, r4
mad oT7.xyz, r1.z, c[32], r2
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
