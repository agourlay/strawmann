push   rbp
mov    rbp,rsp
cmp    rdx,0x20
jae    1003679 <sm_dot_f32+0x29>
xorps  xmm1,xmm1
xor    ecx,ecx
xorps  xmm4,xmm4
xorps  xmm0,xmm0
xorps  xmm2,xmm2
xorps  xmm6,xmm6
xorps  xmm7,xmm7
xorps  xmm5,xmm5
xorps  xmm3,xmm3
jmp    1003752 <sm_dot_f32+0x102>
xorps  xmm3,xmm3
xor    eax,eax
xorps  xmm5,xmm5
xorps  xmm7,xmm7
xorps  xmm6,xmm6
xorps  xmm2,xmm2
xorps  xmm0,xmm0
xorps  xmm4,xmm4
xorps  xmm1,xmm1
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax]
movups xmm8,XMMWORD PTR [rdi+rax*4]
movups xmm9,XMMWORD PTR [rdi+rax*4+0x10]
movups xmm10,XMMWORD PTR [rdi+rax*4+0x20]
movups xmm11,XMMWORD PTR [rdi+rax*4+0x30]
movups xmm12,XMMWORD PTR [rsi+rax*4]
mulps  xmm12,xmm8
addps  xmm7,xmm12
movups xmm8,XMMWORD PTR [rsi+rax*4+0x10]
mulps  xmm8,xmm9
addps  xmm6,xmm8
movups xmm8,XMMWORD PTR [rsi+rax*4+0x20]
mulps  xmm8,xmm10
addps  xmm2,xmm8
movups xmm8,XMMWORD PTR [rsi+rax*4+0x30]
mulps  xmm8,xmm11
addps  xmm0,xmm8
movups xmm8,XMMWORD PTR [rdi+rax*4+0x40]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x40]
mulps  xmm9,xmm8
addps  xmm4,xmm9
movups xmm8,XMMWORD PTR [rdi+rax*4+0x50]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x50]
mulps  xmm9,xmm8
addps  xmm1,xmm9
movups xmm8,XMMWORD PTR [rdi+rax*4+0x60]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x60]
mulps  xmm9,xmm8
addps  xmm5,xmm9
movups xmm8,XMMWORD PTR [rdi+rax*4+0x70]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x70]
mulps  xmm9,xmm8
addps  xmm3,xmm9
lea    rcx,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,rcx
jbe    10036a0 <sm_dot_f32+0x50>
mov    rax,rcx
or     rax,0x4
cmp    rax,rdx
jbe    1003770 <sm_dot_f32+0x120>
mov    rax,rcx
jmp    1003792 <sm_dot_f32+0x142>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax]
movups xmm8,XMMWORD PTR [rdi+rcx*4]
movups xmm9,XMMWORD PTR [rsi+rcx*4]
mulps  xmm9,xmm8
addps  xmm7,xmm9
lea    rax,[rcx+0x4]
add    rcx,0x8
cmp    rcx,rdx
mov    rcx,rax
jbe    1003770 <sm_dot_f32+0x120>
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
jae    1003837 <sm_dot_f32+0x1e7>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x3
je     10037e8 <sm_dot_f32+0x198>
movss  xmm1,DWORD PTR [rsi+rax*4]
mulss  xmm1,DWORD PTR [rdi+rax*4]
addss  xmm0,xmm1
add    rax,0x1
add    r8,0xffffffffffffffff
jne    10037d0 <sm_dot_f32+0x180>
cmp    rcx,0xfffffffffffffffc
ja     1003837 <sm_dot_f32+0x1e7>
xchg   ax,ax
movss  xmm1,DWORD PTR [rsi+rax*4]
movss  xmm2,DWORD PTR [rsi+rax*4+0x4]
mulss  xmm1,DWORD PTR [rdi+rax*4]
mulss  xmm2,DWORD PTR [rdi+rax*4+0x4]
addss  xmm1,xmm0
movss  xmm3,DWORD PTR [rsi+rax*4+0x8]
mulss  xmm3,DWORD PTR [rdi+rax*4+0x8]
addss  xmm3,xmm2
movss  xmm0,DWORD PTR [rsi+rax*4+0xc]
mulss  xmm0,DWORD PTR [rdi+rax*4+0xc]
addss  xmm3,xmm1
addss  xmm0,xmm3
add    rax,0x4
cmp    rdx,rax
jne    10037f0 <sm_dot_f32+0x1a0>
pop    rbp
ret
int3
int3
int3
int3
int3
int3
int3
