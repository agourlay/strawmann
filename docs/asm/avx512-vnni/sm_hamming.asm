cmp    rdx,0x20
jae    1001c8e <sm_hamming+0x1e>
vpxor  xmm0,xmm0,xmm0
xor    r8d,r8d
vpxor  xmm1,xmm1,xmm1
vpxor  xmm3,xmm3,xmm3
vpxor  xmm2,xmm2,xmm2
jmp    1001dd6 <sm_hamming+0x166>
vpxor  xmm4,xmm4,xmm4
xor    eax,eax
vpbroadcastd zmm5,DWORD PTR [rip+0xffffffffffffe742]        # 10003e0 <__anon_3727+0x20>
vbroadcasti32x4 zmm6,XMMWORD PTR [rip+0xffffffffffffe758]        # 1000400 <__anon_4792+0x10>
vpxor  xmm2,xmm2,xmm2
vpxor  xmm3,xmm3,xmm3
vpxor  xmm1,xmm1,xmm1
vpxor  xmm0,xmm0,xmm0
nop    DWORD PTR [rax+rax*1+0x0]
vmovdqu64 zmm7,ZMMWORD PTR [rsi+rax*8]
vmovdqu64 zmm8,ZMMWORD PTR [rsi+rax*8+0x40]
vmovdqu64 zmm9,ZMMWORD PTR [rsi+rax*8+0x80]
vmovdqu64 zmm10,ZMMWORD PTR [rsi+rax*8+0xc0]
vpxorq zmm7,zmm7,ZMMWORD PTR [rdi+rax*8]
vpandq zmm11,zmm7,zmm5
vpshufb zmm11,zmm6,zmm11
vpsrlw zmm7,zmm7,0x4
vpandq zmm7,zmm7,zmm5
vpshufb zmm7,zmm6,zmm7
vpaddb zmm7,zmm7,zmm11
vpsadbw zmm7,zmm7,zmm4
vpaddq zmm1,zmm7,zmm1
vpxorq zmm7,zmm8,ZMMWORD PTR [rdi+rax*8+0x40]
vpandq zmm8,zmm7,zmm5
vpshufb zmm8,zmm6,zmm8
vpsrlw zmm7,zmm7,0x4
vpandq zmm7,zmm7,zmm5
vpshufb zmm7,zmm6,zmm7
vpaddb zmm7,zmm7,zmm8
vpsadbw zmm7,zmm7,zmm4
vpaddq zmm0,zmm7,zmm0
vpxorq zmm7,zmm9,ZMMWORD PTR [rdi+rax*8+0x80]
vpandq zmm8,zmm7,zmm5
vpshufb zmm8,zmm6,zmm8
vpsrlw zmm7,zmm7,0x4
vpandq zmm7,zmm7,zmm5
vpshufb zmm7,zmm6,zmm7
vpaddb zmm7,zmm7,zmm8
vpsadbw zmm7,zmm7,zmm4
vpxorq zmm8,zmm10,ZMMWORD PTR [rdi+rax*8+0xc0]
vpaddq zmm3,zmm7,zmm3
vpandq zmm7,zmm8,zmm5
vpshufb zmm7,zmm6,zmm7
vpsrlw zmm8,zmm8,0x4
vpandq zmm8,zmm8,zmm5
vpshufb zmm8,zmm6,zmm8
vpaddb zmm7,zmm8,zmm7
vpsadbw zmm7,zmm7,zmm4
vpaddq zmm2,zmm7,zmm2
lea    r8,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,r8
jbe    1001cc0 <sm_hamming+0x50>
mov    rax,r8
or     rax,0x8
cmp    rax,rdx
jbe    1001de7 <sm_hamming+0x177>
mov    rcx,r8
jmp    1001e5f <sm_hamming+0x1ef>
push   rbp
mov    rbp,rsp
vpbroadcastd zmm4,DWORD PTR [rip+0xffffffffffffe5eb]        # 10003e0 <__anon_3727+0x20>
vbroadcasti32x4 zmm5,XMMWORD PTR [rip+0xffffffffffffe601]        # 1000400 <__anon_4792+0x10>
vpxor  xmm6,xmm6,xmm6
pop    rbp
data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovdqu64 zmm7,ZMMWORD PTR [rsi+r8*8]
vpxorq zmm7,zmm7,ZMMWORD PTR [rdi+r8*8]
vpandq zmm8,zmm7,zmm4
vpshufb zmm8,zmm5,zmm8
vpsrlw zmm7,zmm7,0x4
vpandq zmm7,zmm7,zmm4
vpshufb zmm7,zmm5,zmm7
vpaddb zmm7,zmm7,zmm8
vpsadbw zmm7,zmm7,zmm6
vpaddq zmm1,zmm7,zmm1
lea    rcx,[r8+0x8]
add    r8,0x10
cmp    r8,rdx
mov    r8,rcx
jbe    1001e10 <sm_hamming+0x1a0>
vpaddq zmm0,zmm3,zmm0
vpaddq zmm0,zmm0,zmm2
vpaddq zmm0,zmm0,zmm1
vextracti64x4 ymm1,zmm0,0x1
vpaddq zmm0,zmm0,zmm1
vextracti128 xmm1,ymm0,0x1
vpaddq xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddq xmm0,xmm0,xmm1
vmovq  rax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1001f85 <sm_hamming+0x315>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1001ec8 <sm_hamming+0x258>
xchg   ax,ax
mov    r10,QWORD PTR [rsi+rcx*8]
xor    r10,QWORD PTR [rdi+rcx*8]
popcnt r10,r10
add    rax,r10
inc    rcx
dec    r9
jne    1001eb0 <sm_hamming+0x240>
cmp    r8,0xfffffffffffffff8
ja     1001f85 <sm_hamming+0x315>
data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
mov    r8,QWORD PTR [rsi+rcx*8]
mov    r9,QWORD PTR [rsi+rcx*8+0x8]
xor    r8,QWORD PTR [rdi+rcx*8]
popcnt r8,r8
add    r8,rax
xor    r9,QWORD PTR [rdi+rcx*8+0x8]
xor    eax,eax
popcnt rax,r9
mov    r9,QWORD PTR [rsi+rcx*8+0x10]
xor    r9,QWORD PTR [rdi+rcx*8+0x10]
popcnt r9,r9
add    r9,rax
add    r9,r8
mov    rax,QWORD PTR [rsi+rcx*8+0x18]
xor    rax,QWORD PTR [rdi+rcx*8+0x18]
popcnt rax,rax
mov    r8,QWORD PTR [rsi+rcx*8+0x20]
xor    r8,QWORD PTR [rdi+rcx*8+0x20]
popcnt r8,r8
add    r8,rax
mov    rax,QWORD PTR [rsi+rcx*8+0x28]
xor    rax,QWORD PTR [rdi+rcx*8+0x28]
xor    r10d,r10d
popcnt r10,rax
add    r10,r8
add    r10,r9
mov    rax,QWORD PTR [rsi+rcx*8+0x30]
xor    rax,QWORD PTR [rdi+rcx*8+0x30]
mov    r8,QWORD PTR [rsi+rcx*8+0x38]
xor    r8,QWORD PTR [rdi+rcx*8+0x38]
xor    r9d,r9d
popcnt r9,rax
xor    eax,eax
popcnt rax,r8
add    rax,r9
add    rax,r10
add    rcx,0x8
cmp    rdx,rcx
jne    1001ee0 <sm_hamming+0x270>
vzeroupper
ret
nop    DWORD PTR [rax+0x0]
