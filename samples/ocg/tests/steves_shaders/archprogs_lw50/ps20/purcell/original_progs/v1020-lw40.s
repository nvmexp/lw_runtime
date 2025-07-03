vs_2_0
def c[1], 0.816497, -0.408248, 0.707107, -0.707107
def c[3], 0.577350, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_tangent0 v4
mov r0.w, c[0].y
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r1.y, r0, c[10]
mad oFog, -r1.y, c[16].w, c[16].x
add r2.xyz, -r0, c[29]
mov oPos.z, r1.y
dp3 r1.z, r2, r2
add oT3.xyz, -r0, c[2]
rsq r0.w, r1.z
mul r6.xyz, r2, r0.w
dst r1.xy, r1.z, r0.w
dp3 r2.x, -r6, c[28]
dp3 r0.x, c[31], r1
add r1.w, r2.x, -c[30].z
rcp r0.w, r0.x
mul r1.w, r1.w, c[30].w
dp3 r7.x, v1, c[42]
dp3 r8.x, v4, c[42]
dp3 r7.y, v1, c[43]
dp3 r7.z, v1, c[44]
dp3 r8.y, v4, c[43]
dp3 r8.z, v4, c[44]
max r2.w, r1.w, c[0].x
mul r0.xyz, r7.zxyw, r8.yzxw
pow r1.w, r2.w, c[30].x
mad r0.xyz, r7.yzxw, r8.zxyw, -r0
min r1.w, r1.w, c[0].y
mul r0.xyz, r0, v4.w
mul r1.xyz, r1.w, c[27]
mul r2.xy, r0.yzzw, c[1].zwzw
mul r1.xyz, r0.w, r1
add r5.y, r2.y, r2.x
mul r3.xy, r8.yzzw, c[1].zwzw
mul r2.xy, r7.yzzw, c[1].zwzw
add r5.x, r3.y, r3.x
add r5.z, r2.y, r2.x
dp3 r2.y, r5, r6
dp3 r4.x, c[1].xyyw, r8
dp3 r4.z, c[1].xyyw, r7
dp3 r4.y, c[1].xyyw, r0
dp3 r2.x, r4, r6
dp3 r3.x, c[3].x, r8
mov oT4.x, r8.x
mov oT5.x, r8.y
mov oT6.x, r8.z
dp3 r3.z, c[3].x, r7
mov oT4.z, r7.x
mov oT5.z, r7.y
mov oT6.z, r7.z
dp3 r3.y, c[3].x, r0
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
