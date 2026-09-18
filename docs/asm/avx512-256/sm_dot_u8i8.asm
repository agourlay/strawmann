push   rbp
mov    rbp,rsp
cmp    rdx,0x80
jae    1002ac4 <sm_dot_u8i8+0x34>
vpxor  xmm1,xmm1,xmm1
xor    eax,eax
vpxor  xmm6,xmm6,xmm6
vpxor  xmm4,xmm4,xmm4
vpxor  xmm2,xmm2,xmm2
vpxor  xmm7,xmm7,xmm7
vpxor  xmm5,xmm5,xmm5
vpxor  xmm0,xmm0,xmm0
vpxor  xmm3,xmm3,xmm3
jmp    1002be8 <sm_dot_u8i8+0x158>
vpxor  xmm0,xmm0,xmm0
xor    ecx,ecx
vpxor  xmm3,xmm3,xmm3
vpxor  xmm7,xmm7,xmm7
vpxor  xmm5,xmm5,xmm5
vpxor  xmm4,xmm4,xmm4
vpxor  xmm2,xmm2,xmm2
vpxor  xmm1,xmm1,xmm1
vpxor  xmm6,xmm6,xmm6
cs nop WORD PTR [rax+rax*1+0x0]
vpmovzxbd zmm8,XMMWORD PTR [rdi+rcx*1+0x10]
vpmovzxbd zmm9,XMMWORD PTR [rdi+rcx*1]
vpmovsxbd zmm10,XMMWORD PTR [rsi+rcx*1+0x10]
vpmaddwd zmm8,zmm10,zmm8
vpmovsxbd zmm10,XMMWORD PTR [rsi+rcx*1]
vpaddd zmm2,zmm8,zmm2
vpmaddwd zmm8,zmm10,zmm9
vpmovzxbd zmm9,XMMWORD PTR [rdi+rcx*1+0x30]
vpmovzxbd zmm10,XMMWORD PTR [rdi+rcx*1+0x20]
vpaddd zmm4,zmm8,zmm4
vpmovsxbd zmm8,XMMWORD PTR [rsi+rcx*1+0x30]
vpmaddwd zmm8,zmm8,zmm9
vpmovsxbd zmm9,XMMWORD PTR [rsi+rcx*1+0x20]
vpaddd zmm6,zmm8,zmm6
vpmaddwd zmm8,zmm9,zmm10
vpmovzxbd zmm9,XMMWORD PTR [rdi+rcx*1+0x50]
vpmovzxbd zmm10,XMMWORD PTR [rdi+rcx*1+0x40]
vpaddd zmm1,zmm8,zmm1
vpmovsxbd zmm8,XMMWORD PTR [rsi+rcx*1+0x50]
vpmaddwd zmm8,zmm8,zmm9
vpmovsxbd zmm9,XMMWORD PTR [rsi+rcx*1+0x40]
vpaddd zmm5,zmm8,zmm5
vpmaddwd zmm8,zmm9,zmm10
vpmovzxbd zmm9,XMMWORD PTR [rdi+rcx*1+0x70]
vpmovzxbd zmm10,XMMWORD PTR [rdi+rcx*1+0x60]
vpaddd zmm7,zmm8,zmm7
vpmovsxbd zmm8,XMMWORD PTR [rsi+rcx*1+0x70]
vpmaddwd zmm8,zmm8,zmm9
vpmovsxbd zmm9,XMMWORD PTR [rsi+rcx*1+0x60]
vpaddd zmm3,zmm8,zmm3
vpmaddwd zmm8,zmm9,zmm10
vpaddd zmm0,zmm8,zmm0
lea    rax,[rcx+0x80]
add    rcx,0x100
cmp    rcx,rdx
mov    rcx,rax
jbe    1002af0 <sm_dot_u8i8+0x60>
mov    rcx,rax
or     rcx,0x20
cmp    rcx,rdx
jbe    1002c00 <sm_dot_u8i8+0x170>
mov    rcx,rax
jmp    1002c46 <sm_dot_u8i8+0x1b6>
nop    DWORD PTR [rax+0x0]
vpmovzxbd zmm8,XMMWORD PTR [rdi+rax*1+0x10]
vpmovzxbd zmm9,XMMWORD PTR [rdi+rax*1]
vpmovsxbd zmm10,XMMWORD PTR [rsi+rax*1+0x10]
vpmaddwd zmm8,zmm10,zmm8
vpmovsxbd zmm10,XMMWORD PTR [rsi+rax*1]
vpaddd zmm2,zmm8,zmm2
vpmaddwd zmm8,zmm10,zmm9
vpaddd zmm4,zmm8,zmm4
lea    rcx,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,rcx
jbe    1002c00 <sm_dot_u8i8+0x170>
vpaddd zmm1,zmm7,zmm1
vpaddd zmm0,zmm1,zmm0
vpaddd zmm0,zmm0,zmm4
vpaddd zmm1,zmm5,zmm6
vpaddd zmm1,zmm1,zmm3
vpaddd zmm1,zmm1,zmm2
vpaddd zmm0,zmm0,zmm1
vextracti64x4 ymm1,zmm0,0x1
vpaddd zmm0,zmm0,zmm1
vextracti128 xmm1,ymm0,0x1
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0x55
vpaddd xmm0,xmm0,xmm1
vmovd  eax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1002d8c <sm_dot_u8i8+0x2fc>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1002cd9 <sm_dot_u8i8+0x249>
data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movsx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
inc    rcx
dec    r9
jne    1002cc0 <sm_dot_u8i8+0x230>
cmp    r8,0xfffffffffffffff8
ja     1002d8c <sm_dot_u8i8+0x2fc>
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
jne    1002cf0 <sm_dot_u8i8+0x260>
pop    rbp
vzeroupper
ret
data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
