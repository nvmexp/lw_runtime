vs_2_0
def c[1], 0.816497, -0.408248, 0.707107, -0.707107
def c[3], 0.577350, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v4
mov r1.w, c[0].y
dp4 r1.x, v0, c[42]
dp4 r1.y, v0, c[43]
dp4 r1.z, v0, c[44]
dp4 oPos.x, r1, c[8]
dp4 oPos.y, r1, c[9]
dp4 oPos.w, r1, c[11]
dp4 r0.y, r1, c[10]
mad oFog, -r0.y, c[16].w, c[16].x
add r3.xyz, -r1, c[34]
mov oPos.z, r0.y
dp3 r0.z, r3, r3
add oT3.xyz, -r1, c[2]
rsq r0.w, r0.z
dst r0.xy, r0.z, r0.w
add r2.xyz, -r1, c[29]
dp4 r1.w, c[36], r0
mul r8.xyz, r3, r0.w
dp3 r0.z, r2, r2
rcp r1.w, r1.w
rsq r0.w, r0.z
mul r1.xyz, r1.w, c[32]
mul r7.xyz, r2, r0.w
dp3 r2.x, -r7, c[28]
dst r0.xy, r0.z, r0.w
add r0.w, r2.x, -c[30].z
dp3 r0.x, c[31], r0
mul r1.w, r0.w, c[30].w
rcp r0.w, r0.x
max r2.w, r1.w, c[0].x
dp3 r2.x, v1, c[42]
dp3 r10.x, v4, c[42]
dp3 r2.y, v1, c[43]
dp3 r2.z, v1, c[44]
dp3 r10.y, v4, c[43]
dp3 r10.z, v4, c[44]
pow r1.w, r2.w, c[30].x
mul r0.xyz, r2.zxyw, r10.yzxw
min r1.w, r1.w, c[0].y
mad r0.xyz, r2.yzxw, r10.zxyw, -r0
mul r3.xyz, r1.w, c[27]
mul r0.xyz, r0, v4.w
mul r3.xyz, r0.w, r3
mul r4.xy, r0.yzzw, c[1].zwzw
add r6.y, r4.y, r4.x
mul r4.xy, r10.yzzw, c[1].zwzw
add r6.x, r4.y, r4.x
mul r4.xy, r2.yzzw, c[1].zwzw
add r6.z, r4.y, r4.x
dp3 r9.x, c[1].xyyw, r10
dp3 r9.z, c[1].xyyw, r2
dp3 r9.y, c[1].xyyw, r0
dp3 r4.y, r6, r7
dp3 r4.x, r9, r7
dp3 r5.x, c[3].x, r10
mov oT4.x, r10.x
mov oT5.x, r10.y
mov oT6.x, r10.z
dp3 r5.z, c[3].x, r2
mov oT4.z, r2.x
mov oT5.z, r2.y
mov oT6.z, r2.z
dp3 r5.y, c[3].x, r0
slt r2.xyz, r9, c[0].x
mova a0.xyz, r2
mul r2.xyz, r9, r9
dp3 r4.z, r5, r7
mul r7.xyz, r2.y, c[a0.y+23]
max r4.xyz, r4, c[0].x
mad r7.xyz, r2.x, c[a0.x+21], r7
dp3 r2.x, r9, r8
mad r7.xyz, r2.z, c[a0.z+25], r7
mad r7.xyz, r3, r4.x, r7
dp3 r2.y, r6, r8
dp3 r2.z, r5, r8
slt r8.xyz, r6, c[0].x
mul r6.xyz, r6, r6
mova a0.xyw, r8.xyz
max r2.xyz, r2, c[0].x
mul r8.xyz, r6.y, c[a0.y+23]
mad r8.xyz, r6.x, c[a0.x+21], r8
slt r9.xyz, r5, c[0].x
mul r5.xyz, r5, r5
mova a0.xyz, r9
mad oD0.xyz, r1, r2.x, r7
mul r7.xyz, r5.y, c[a0.y+23]
mad r6.xyz, r6.z, c[a0.w+25], r8
mad r7.xyz, r5.x, c[a0.x+21], r7
mad r6.xyz, r3, r4.y, r6
mad r5.xyz, r5.z, c[a0.z+25], r7
mad oD1.xyz, r1, r2.y, r6
mad r3.xyz, r3, r4.z, r5
mad oT7.xyz, r1, r2.z, r3
mul r1.xy, v2, c[90]
add r0.w, r1.y, r1.x
mul r1.xy, v2, c[91]
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
