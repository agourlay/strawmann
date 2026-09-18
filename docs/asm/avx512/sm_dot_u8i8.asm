push   rbp
mov    rbp,rsp
cmp    rdx,0x100
jae    1002d9c <sm_dot_u8i8+0x5c>
vpxor  xmm1,xmm1,xmm1
xor    eax,eax
vpxor  xmm13,xmm13,xmm13
vpxor  xmm7,xmm7,xmm7
vpxor  xmm15,xmm15,xmm15
vpxor  xmm2,xmm2,xmm2
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm10,xmm10,xmm10
vpxor  xmm5,xmm5,xmm5
vpxor  xmm12,xmm12,xmm12
vpxor  xmm6,xmm6,xmm6
vpxor  xmm14,xmm14,xmm14
vpxor  xmm0,xmm0,xmm0
vpxor  xmm9,xmm9,xmm9
vpxor  xmm4,xmm4,xmm4
vpxor  xmm11,xmm11,xmm11
jmp    1002fc8 <sm_dot_u8i8+0x288>
vpxor  xmm0,xmm0,xmm0
xor    ecx,ecx
vpxor  xmm9,xmm9,xmm9
vpxor  xmm4,xmm4,xmm4
vpxor  xmm11,xmm11,xmm11
vpxor  xmm5,xmm5,xmm5
vpxor  xmm12,xmm12,xmm12
vpxor  xmm6,xmm6,xmm6
vpxor  xmm14,xmm14,xmm14
vpxor  xmm2,xmm2,xmm2
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm10,xmm10,xmm10
vpxor  xmm1,xmm1,xmm1
vpxor  xmm13,xmm13,xmm13
vpxor  xmm7,xmm7,xmm7
vpxor  xmm15,xmm15,xmm15
cs nop WORD PTR [rax+rax*1+0x0]
vpmovzxbd zmm16,XMMWORD PTR [rdi+rcx*1+0x20]
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0x10]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rcx*1+0x30]
vpmovsxbd zmm20,XMMWORD PTR [rsi+rcx*1+0x20]
vpmaddwd zmm16,zmm20,zmm16
vpaddd zmm3,zmm16,zmm3
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x10]
vpmaddwd zmm16,zmm16,zmm17
vpaddd zmm8,zmm16,zmm8
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1]
vpmaddwd zmm16,zmm16,zmm18
vpmovsxbd zmm17,XMMWORD PTR [rsi+rcx*1+0x30]
vpaddd zmm2,zmm16,zmm2
vpmaddwd zmm16,zmm17,zmm19
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0x60]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1+0x50]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rcx*1+0x40]
vpmovzxbd zmm20,XMMWORD PTR [rdi+rcx*1+0x70]
vpmovsxbd zmm21,XMMWORD PTR [rsi+rcx*1+0x60]
vpaddd zmm10,zmm16,zmm10
vpmaddwd zmm16,zmm21,zmm17
vpaddd zmm7,zmm16,zmm7
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x50]
vpmaddwd zmm16,zmm16,zmm18
vpaddd zmm13,zmm16,zmm13
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x40]
vpmaddwd zmm16,zmm16,zmm19
vpmovsxbd zmm17,XMMWORD PTR [rsi+rcx*1+0x70]
vpaddd zmm1,zmm16,zmm1
vpmaddwd zmm16,zmm17,zmm20
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0xa0]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1+0x90]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rcx*1+0x80]
vpmovzxbd zmm20,XMMWORD PTR [rdi+rcx*1+0xb0]
vpmovsxbd zmm21,XMMWORD PTR [rsi+rcx*1+0xa0]
vpaddd zmm15,zmm16,zmm15
vpmaddwd zmm16,zmm21,zmm17
vpaddd zmm6,zmm16,zmm6
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x90]
vpmaddwd zmm16,zmm16,zmm18
vpaddd zmm12,zmm16,zmm12
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x80]
vpmaddwd zmm16,zmm16,zmm19
vpmovsxbd zmm17,XMMWORD PTR [rsi+rcx*1+0xb0]
vpaddd zmm5,zmm16,zmm5
vpmaddwd zmm16,zmm17,zmm20
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0xe0]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1+0xd0]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rcx*1+0xc0]
vpmovzxbd zmm20,XMMWORD PTR [rdi+rcx*1+0xf0]
vpmovsxbd zmm21,XMMWORD PTR [rsi+rcx*1+0xe0]
vpaddd zmm14,zmm16,zmm14
vpmaddwd zmm16,zmm21,zmm17
vpaddd zmm4,zmm16,zmm4
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1+0xd0]
vpmaddwd zmm16,zmm16,zmm18
vpaddd zmm9,zmm16,zmm9
vpmovsxbd zmm16,XMMWORD PTR [rsi+rcx*1+0xc0]
vpmaddwd zmm16,zmm16,zmm19
vpmovsxbd zmm17,XMMWORD PTR [rsi+rcx*1+0xf0]
vpaddd zmm0,zmm16,zmm0
vpmaddwd zmm16,zmm17,zmm20
vpaddd zmm11,zmm16,zmm11
lea    rax,[rcx+0x100]
add    rcx,0x200
cmp    rcx,rdx
mov    rcx,rax
jbe    1002df0 <sm_dot_u8i8+0xb0>
mov    rcx,rax
or     rcx,0x40
cmp    rcx,rdx
jbe    1002fe0 <sm_dot_u8i8+0x2a0>
mov    rcx,rax
jmp    100305e <sm_dot_u8i8+0x31e>
nop    DWORD PTR [rax+0x0]
vpmovzxbd zmm16,XMMWORD PTR [rdi+rax*1+0x20]
vpmovzxbd zmm17,XMMWORD PTR [rdi+rax*1+0x10]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rax*1]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rax*1+0x30]
vpmovsxbd zmm20,XMMWORD PTR [rsi+rax*1+0x20]
vpmaddwd zmm16,zmm20,zmm16
vpaddd zmm3,zmm16,zmm3
vpmovsxbd zmm16,XMMWORD PTR [rsi+rax*1+0x10]
vpmaddwd zmm16,zmm16,zmm17
vpaddd zmm8,zmm16,zmm8
vpmovsxbd zmm16,XMMWORD PTR [rsi+rax*1]
vpmaddwd zmm16,zmm16,zmm18
vpmovsxbd zmm17,XMMWORD PTR [rsi+rax*1+0x30]
vpaddd zmm2,zmm16,zmm2
vpmaddwd zmm16,zmm17,zmm19
vpaddd zmm10,zmm16,zmm10
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1002fe0 <sm_dot_u8i8+0x2a0>
vpaddd zmm12,zmm12,zmm13
vpaddd zmm9,zmm12,zmm9
vpaddd zmm8,zmm9,zmm8
vpaddd zmm9,zmm14,zmm15
vpaddd zmm9,zmm9,zmm11
vpaddd zmm9,zmm9,zmm10
vpaddd zmm8,zmm8,zmm9
vpaddd zmm1,zmm5,zmm1
vpaddd zmm0,zmm1,zmm0
vpaddd zmm0,zmm0,zmm2
vpaddd zmm1,zmm6,zmm7
vpaddd zmm1,zmm1,zmm4
vpaddd zmm1,zmm1,zmm3
vpaddd zmm0,zmm0,zmm1
vpaddd zmm0,zmm0,zmm8
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
jae    10031cc <sm_dot_u8i8+0x48c>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1003119 <sm_dot_u8i8+0x3d9>
nop    DWORD PTR [rax]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movsx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
inc    rcx
dec    r9
jne    1003100 <sm_dot_u8i8+0x3c0>
cmp    r8,0xfffffffffffffff8
ja     10031cc <sm_dot_u8i8+0x48c>
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
jne    1003130 <sm_dot_u8i8+0x3f0>
pop    rbp
vzeroupper
ret
data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
