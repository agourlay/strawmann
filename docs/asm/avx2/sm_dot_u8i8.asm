push   rbp
mov    rbp,rsp
and    rsp,0xffffffffffffffe0
sub    rsp,0x140
cmp    rdx,0x80
jae    1002d10 <sm_dot_u8i8+0x70>
vpxor  xmm0,xmm0,xmm0
vmovdqa YMMWORD PTR [rsp+0x60],ymm0
xor    eax,eax
vpxor  xmm11,xmm11,xmm11
vpxor  xmm9,xmm9,xmm9
vpxor  xmm2,xmm2,xmm2
vpxor  xmm15,xmm15,xmm15
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm10,xmm10,xmm10
vmovdqa YMMWORD PTR [rsp+0x40],ymm0
vpxor  xmm14,xmm14,xmm14
vpxor  xmm7,xmm7,xmm7
vpxor  xmm12,xmm12,xmm12
vpxor  xmm4,xmm4,xmm4
vmovdqa YMMWORD PTR [rsp+0x20],ymm0
vpxor  xmm6,xmm6,xmm6
vpxor  xmm1,xmm1,xmm1
jmp    1002f6d <sm_dot_u8i8+0x2cd>
vpxor  xmm4,xmm4,xmm4
xor    ecx,ecx
vpxor  xmm0,xmm0,xmm0
vmovdqa YMMWORD PTR [rsp+0x20],ymm0
vpxor  xmm6,xmm6,xmm6
vpxor  xmm1,xmm1,xmm1
vmovdqa YMMWORD PTR [rsp+0x40],ymm0
vpxor  xmm14,xmm14,xmm14
vpxor  xmm7,xmm7,xmm7
vpxor  xmm12,xmm12,xmm12
vpxor  xmm15,xmm15,xmm15
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm10,xmm10,xmm10
vmovdqa YMMWORD PTR [rsp+0x60],ymm0
vpxor  xmm11,xmm11,xmm11
vpxor  xmm9,xmm9,xmm9
vpxor  xmm2,xmm2,xmm2
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovdqa YMMWORD PTR [rsp],ymm1
vmovdqa YMMWORD PTR [rsp+0x80],ymm6
vmovdqa YMMWORD PTR [rsp+0xa0],ymm4
vpmovzxbd ymm0,QWORD PTR [rdi+rcx*1+0x10]
vpmovzxbd ymm1,QWORD PTR [rdi+rcx*1+0x8]
vpmovzxbd ymm4,QWORD PTR [rdi+rcx*1]
vpmovzxbd ymm5,QWORD PTR [rdi+rcx*1+0x18]
vpmovsxbd ymm6,QWORD PTR [rsi+rcx*1+0x10]
vpmaddwd ymm0,ymm6,ymm0
vpaddd ymm3,ymm0,ymm3
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x8]
vpmaddwd ymm0,ymm0,ymm1
vpaddd ymm8,ymm8,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1]
vpmaddwd ymm0,ymm0,ymm4
vpaddd ymm15,ymm15,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x18]
vpmaddwd ymm0,ymm0,ymm5
vpmovzxbd ymm1,QWORD PTR [rdi+rcx*1+0x30]
vpmovzxbd ymm4,QWORD PTR [rdi+rcx*1+0x28]
vpmovzxbd ymm5,QWORD PTR [rdi+rcx*1+0x20]
vpmovzxbd ymm6,QWORD PTR [rdi+rcx*1+0x38]
vpaddd ymm10,ymm10,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x30]
vpmaddwd ymm0,ymm0,ymm1
vpaddd ymm9,ymm9,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x28]
vpmaddwd ymm0,ymm0,ymm4
vpaddd ymm11,ymm11,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x20]
vpmaddwd ymm0,ymm0,ymm5
vmovdqa ymm1,YMMWORD PTR [rsp+0x60]
vpaddd ymm1,ymm0,ymm1
vmovdqa YMMWORD PTR [rsp+0x60],ymm1
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x38]
vpmaddwd ymm0,ymm0,ymm6
vpmovzxbd ymm1,QWORD PTR [rdi+rcx*1+0x50]
vpmovzxbd ymm4,QWORD PTR [rdi+rcx*1+0x48]
vpmovzxbd ymm5,QWORD PTR [rdi+rcx*1+0x40]
vpmovzxbd ymm6,QWORD PTR [rdi+rcx*1+0x58]
vpaddd ymm2,ymm0,ymm2
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x50]
vpmaddwd ymm0,ymm0,ymm1
vpaddd ymm7,ymm0,ymm7
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x48]
vpmaddwd ymm0,ymm0,ymm4
vpaddd ymm14,ymm14,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x40]
vpmaddwd ymm0,ymm0,ymm5
vmovdqa ymm1,YMMWORD PTR [rsp+0x40]
vpaddd ymm1,ymm0,ymm1
vmovdqa YMMWORD PTR [rsp+0x40],ymm1
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x58]
vpmaddwd ymm0,ymm0,ymm6
vpmovzxbd ymm1,QWORD PTR [rdi+rcx*1+0x70]
vpmovzxbd ymm4,QWORD PTR [rdi+rcx*1+0x68]
vpmovzxbd ymm5,QWORD PTR [rdi+rcx*1+0x60]
vpmovsxbd ymm6,QWORD PTR [rsi+rcx*1+0x70]
vpaddd ymm13,ymm12,ymm0
vpmaddwd ymm0,ymm6,ymm1
vmovdqa ymm6,YMMWORD PTR [rsp+0x80]
vpmovsxbd ymm1,QWORD PTR [rsi+rcx*1+0x68]
vpaddd ymm6,ymm0,ymm6
vpmaddwd ymm0,ymm1,ymm4
vmovdqa ymm4,YMMWORD PTR [rsp+0xa0]
vpmovsxbd ymm1,QWORD PTR [rsi+rcx*1+0x60]
vmovdqa ymm12,ymm9
vmovdqa ymm9,ymm7
vmovdqa ymm7,ymm14
vmovdqa ymm14,ymm11
vmovdqa ymm11,YMMWORD PTR [rsp+0x20]
vpaddd ymm11,ymm11,ymm0
vmovdqa YMMWORD PTR [rsp+0x20],ymm11
vmovdqa ymm11,ymm14
vmovdqa ymm14,ymm7
vmovdqa ymm7,ymm9
vmovdqa ymm9,ymm12
vmovdqa ymm12,ymm13
vpmaddwd ymm0,ymm1,ymm5
vpmovzxbd ymm1,QWORD PTR [rdi+rcx*1+0x78]
vpaddd ymm4,ymm0,ymm4
vpmovsxbd ymm0,QWORD PTR [rsi+rcx*1+0x78]
vpmaddwd ymm0,ymm0,ymm1
vmovdqa ymm1,YMMWORD PTR [rsp]
vpaddd ymm1,ymm0,ymm1
vmovdqa YMMWORD PTR [rsp],ymm1
vmovdqa ymm1,YMMWORD PTR [rsp]
lea    rax,[rcx+0x80]
add    rcx,0x100
cmp    rcx,rdx
mov    rcx,rax
jbe    1002d70 <sm_dot_u8i8+0xd0>
vmovdqa YMMWORD PTR [rsp+0xc0],ymm2
vmovdqa YMMWORD PTR [rsp+0xe0],ymm12
vmovdqa YMMWORD PTR [rsp+0x100],ymm11
vmovdqa YMMWORD PTR [rsp],ymm14
vmovdqa ymm14,YMMWORD PTR [rsp+0x20]
vmovdqa ymm13,ymm9
vmovdqa ymm12,ymm7
vmovdqa ymm9,YMMWORD PTR [rsp+0x40]
vmovdqa YMMWORD PTR [rsp+0x80],ymm6
vmovdqa ymm7,YMMWORD PTR [rsp+0x60]
vmovdqa YMMWORD PTR [rsp+0xa0],ymm4
mov    rcx,rax
or     rcx,0x20
vmovdqa ymm2,ymm1
cmp    rcx,rdx
jbe    1002fd4 <sm_dot_u8i8+0x334>
mov    rcx,rax
vmovdqa ymm11,ymm12
jmp    1003046 <sm_dot_u8i8+0x3a6>
vmovdqa ymm11,ymm12
nop    DWORD PTR [rax+0x0]
vpmovzxbd ymm0,QWORD PTR [rdi+rax*1+0x10]
vpmovzxbd ymm1,QWORD PTR [rdi+rax*1+0x8]
vpmovzxbd ymm4,QWORD PTR [rdi+rax*1]
vpmovzxbd ymm5,QWORD PTR [rdi+rax*1+0x18]
vpmovsxbd ymm6,QWORD PTR [rsi+rax*1+0x10]
vpmaddwd ymm0,ymm6,ymm0
vpaddd ymm3,ymm0,ymm3
vpmovsxbd ymm0,QWORD PTR [rsi+rax*1+0x8]
vpmaddwd ymm0,ymm0,ymm1
vpaddd ymm8,ymm8,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rax*1]
vpmaddwd ymm0,ymm0,ymm4
vpaddd ymm15,ymm15,ymm0
vpmovsxbd ymm0,QWORD PTR [rsi+rax*1+0x18]
vpmaddwd ymm0,ymm0,ymm5
vpaddd ymm10,ymm10,ymm0
lea    rcx,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,rcx
jbe    1002fe0 <sm_dot_u8i8+0x340>
vmovdqa ymm0,YMMWORD PTR [rsp+0x100]
vpaddd ymm0,ymm0,YMMWORD PTR [rsp]
vpaddd ymm0,ymm14,ymm0
vpaddd ymm0,ymm8,ymm0
vmovdqa ymm1,YMMWORD PTR [rsp+0xc0]
vpaddd ymm1,ymm1,YMMWORD PTR [rsp+0xe0]
vpaddd ymm1,ymm1,ymm2
vpaddd ymm1,ymm10,ymm1
vpaddd ymm0,ymm0,ymm1
vpaddd ymm1,ymm9,ymm7
vpaddd ymm1,ymm1,YMMWORD PTR [rsp+0xa0]
vpaddd ymm1,ymm15,ymm1
vpaddd ymm2,ymm11,ymm13
vpaddd ymm2,ymm2,YMMWORD PTR [rsp+0x80]
vpaddd ymm2,ymm2,ymm3
vpaddd ymm1,ymm1,ymm2
vpaddd ymm0,ymm1,ymm0
vextracti128 xmm1,ymm0,0x1
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0x55
vpaddd xmm0,xmm0,xmm1
vmovd  eax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    10031ac <sm_dot_u8i8+0x50c>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     10030f9 <sm_dot_u8i8+0x459>
nop    DWORD PTR [rax]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movsx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
inc    rcx
dec    r9
jne    10030e0 <sm_dot_u8i8+0x440>
cmp    r8,0xfffffffffffffff8
ja     10031ac <sm_dot_u8i8+0x50c>
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r8d,BYTE PTR [rdi+rcx*1]
movsx  r9d,BYTE PTR [rsi+rcx*1]
imul   r9d,r8d
add    r9d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x1]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x1]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x2]
movsx  r10d,BYTE PTR [rsi+rcx*1+0x2]
imul   r10d,eax
add    r10d,r8d
add    r10d,r9d
movzx  eax,BYTE PTR [rdi+rcx*1+0x3]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x3]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x4]
movsx  r9d,BYTE PTR [rsi+rcx*1+0x4]
imul   r9d,eax
add    r9d,r8d
movzx  eax,BYTE PTR [rdi+rcx*1+0x5]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x5]
imul   r8d,eax
add    r8d,r9d
add    r8d,r10d
movzx  eax,BYTE PTR [rdi+rcx*1+0x6]
movsx  r9d,BYTE PTR [rsi+rcx*1+0x6]
imul   r9d,eax
movzx  r10d,BYTE PTR [rdi+rcx*1+0x7]
movsx  eax,BYTE PTR [rsi+rcx*1+0x7]
imul   eax,r10d
add    eax,r9d
add    eax,r8d
add    rcx,0x8
cmp    rdx,rcx
jne    1003110 <sm_dot_u8i8+0x470>
mov    rsp,rbp
pop    rbp
vzeroupper
ret
data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
