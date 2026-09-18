push   rbp
mov    rbp,rsp
cmp    rdx,0x20
jae    1002589 <sm_dot_f32u8+0x29>
xorps  xmm1,xmm1
xor    ecx,ecx
xorps  xmm4,xmm4
xorps  xmm0,xmm0
xorps  xmm2,xmm2
xorps  xmm6,xmm6
xorps  xmm7,xmm7
xorps  xmm5,xmm5
xorps  xmm3,xmm3
jmp    10026da <sm_dot_f32u8+0x17a>
pxor   xmm8,xmm8
xor    eax,eax
xorps  xmm3,xmm3
xorps  xmm5,xmm5
xorps  xmm7,xmm7
xorps  xmm6,xmm6
xorps  xmm2,xmm2
xorps  xmm0,xmm0
xorps  xmm4,xmm4
xorps  xmm1,xmm1
nop    DWORD PTR [rax+rax*1+0x0]
movd   xmm9,DWORD PTR [rsi+rax*1]
punpcklbw xmm9,xmm8
punpcklwd xmm9,xmm8
cvtdq2ps xmm9,xmm9
movups xmm10,XMMWORD PTR [rdi+rax*4]
mulps  xmm10,xmm9
addps  xmm7,xmm10
movups xmm9,XMMWORD PTR [rdi+rax*4+0x10]
movups xmm10,XMMWORD PTR [rdi+rax*4+0x20]
movups xmm11,XMMWORD PTR [rdi+rax*4+0x30]
movd   xmm12,DWORD PTR [rsi+rax*1+0x4]
punpcklbw xmm12,xmm8
punpcklwd xmm12,xmm8
cvtdq2ps xmm12,xmm12
mulps  xmm12,xmm9
addps  xmm6,xmm12
movd   xmm9,DWORD PTR [rsi+rax*1+0x8]
punpcklbw xmm9,xmm8
punpcklwd xmm9,xmm8
cvtdq2ps xmm9,xmm9
mulps  xmm9,xmm10
addps  xmm2,xmm9
movd   xmm9,DWORD PTR [rsi+rax*1+0xc]
punpcklbw xmm9,xmm8
punpcklwd xmm9,xmm8
cvtdq2ps xmm9,xmm9
mulps  xmm9,xmm11
addps  xmm0,xmm9
movups xmm9,XMMWORD PTR [rdi+rax*4+0x40]
movd   xmm10,DWORD PTR [rsi+rax*1+0x10]
punpcklbw xmm10,xmm8
punpcklwd xmm10,xmm8
cvtdq2ps xmm10,xmm10
mulps  xmm10,xmm9
addps  xmm4,xmm10
movups xmm9,XMMWORD PTR [rdi+rax*4+0x50]
movd   xmm10,DWORD PTR [rsi+rax*1+0x14]
punpcklbw xmm10,xmm8
punpcklwd xmm10,xmm8
cvtdq2ps xmm10,xmm10
mulps  xmm10,xmm9
addps  xmm1,xmm10
movups xmm9,XMMWORD PTR [rdi+rax*4+0x60]
movd   xmm10,DWORD PTR [rsi+rax*1+0x18]
punpcklbw xmm10,xmm8
punpcklwd xmm10,xmm8
cvtdq2ps xmm10,xmm10
mulps  xmm10,xmm9
addps  xmm5,xmm10
movups xmm9,XMMWORD PTR [rdi+rax*4+0x70]
movd   xmm10,DWORD PTR [rsi+rax*1+0x1c]
punpcklbw xmm10,xmm8
punpcklwd xmm10,xmm8
cvtdq2ps xmm10,xmm10
mulps  xmm10,xmm9
addps  xmm3,xmm10
lea    rcx,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,rcx
jbe    10025b0 <sm_dot_f32u8+0x50>
mov    rax,rcx
or     rax,0x4
cmp    rax,rdx
jbe    10026eb <sm_dot_f32u8+0x18b>
mov    rax,rcx
jmp    1002721 <sm_dot_f32u8+0x1c1>
pxor   xmm8,xmm8
movups xmm9,XMMWORD PTR [rdi+rcx*4]
movd   xmm10,DWORD PTR [rsi+rcx*1]
punpcklbw xmm10,xmm8
punpcklwd xmm10,xmm8
cvtdq2ps xmm10,xmm10
mulps  xmm10,xmm9
addps  xmm7,xmm10
lea    rax,[rcx+0x4]
add    rcx,0x8
cmp    rcx,rdx
mov    rcx,rax
jbe    10026f0 <sm_dot_f32u8+0x190>
addps  xmm4,xmm7
addps  xmm1,xmm6
addps  xmm2,xmm5
addps  xmm2,xmm4
addps  xmm0,xmm3
addps  xmm0,xmm1
addps  xmm0,xmm2
movaps xmm1,xmm0
unpckhpd xmm1,xmm0
addps  xmm1,xmm0
movaps xmm0,xmm1
shufps xmm0,xmm1,0x55
addss  xmm0,xmm1
mov    rcx,rax
sub    rcx,rdx
jae    10027ff <sm_dot_f32u8+0x29f>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x3
je     1002790 <sm_dot_f32u8+0x230>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax]
movzx  r9d,BYTE PTR [rsi+rax*1]
xorps  xmm1,xmm1
cvtsi2ss xmm1,r9d
mulss  xmm1,DWORD PTR [rdi+rax*4]
addss  xmm0,xmm1
add    rax,0x1
add    r8,0xffffffffffffffff
jne    1002770 <sm_dot_f32u8+0x210>
cmp    rcx,0xfffffffffffffffc
ja     10027ff <sm_dot_f32u8+0x29f>
cs nop WORD PTR [rax+rax*1+0x0]
movzx  ecx,BYTE PTR [rsi+rax*1]
xorps  xmm1,xmm1
cvtsi2ss xmm1,ecx
mulss  xmm1,DWORD PTR [rdi+rax*4]
addss  xmm1,xmm0
movzx  ecx,BYTE PTR [rsi+rax*1+0x1]
xorps  xmm2,xmm2
cvtsi2ss xmm2,ecx
movzx  ecx,BYTE PTR [rsi+rax*1+0x2]
xorps  xmm3,xmm3
cvtsi2ss xmm3,ecx
mulss  xmm2,DWORD PTR [rdi+rax*4+0x4]
mulss  xmm3,DWORD PTR [rdi+rax*4+0x8]
movzx  ecx,BYTE PTR [rsi+rax*1+0x3]
xorps  xmm0,xmm0
cvtsi2ss xmm0,ecx
addss  xmm3,xmm2
mulss  xmm0,DWORD PTR [rdi+rax*4+0xc]
addss  xmm3,xmm1
addss  xmm0,xmm3
add    rax,0x4
cmp    rdx,rax
jne    10027a0 <sm_dot_f32u8+0x240>
pop    rbp
ret
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax+rax*1+0x0]
