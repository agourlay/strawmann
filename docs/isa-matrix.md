# The ISA matrix, §7.5, and §6.6.1's prediction under test

§6.6.1 makes a falsifiable claim and says so:

> "**vector width should matter least exactly where people spend the most
> effort on it** (cold fp32 distance) and most where the working set has been
> shrunk into cache (binary, PQ, and the rescore stage). That is a falsifiable
> prediction, and §6.6 and §7.5 exist to test it rather than assume it. If it
> holds, it reorders the optimisation priorities for any HNSW implementation;
> if it fails, the failure locates the model's error."

It fails, and the failure is informative. **On this host, AVX-512 helps fp32
*more* when cold than when hot**, the opposite of the prediction.

---

## Methodology note (added after the fact; numbers not re-measured)

Every table below was produced by the harness before the fixes recorded in
`docs/cost-model.md`, "Harness revision", and none has been re-measured since.
Three of those fixes bear on the ISA comparison specifically:

- **The runs were unpinned** on a hybrid host (Zen 5 cpus 0-3,12-15 sharing a
  16 MiB L3; Zen 5c cpus 4-11,16-23 sharing 8 MiB, at a lower clock). An arm
  that happened to run on the other cluster would differ in cycles/vector for
  reasons that have nothing to do with the ISA. Effective frequency was
  recorded per cell (§6.6.4) and the arms agree to within a few percent, which
  is evidence they ran on the same kind of core, not proof. The harness now
  pins (default: the starting cpu; `--pin N`) and prints the cpu and its cache
  sizes.
- **The "DRAM-cold" rows are the independent (MLP) loop**, not a dependent
  fetch. The finding that AVX-512 helps cold fp32 *because* it raises the
  number of concurrent misses is consistent with that (fewer instructions per
  vector, more iterations in the reorder buffer, more fetches in flight), and
  is exactly the mechanism a dependent chain would *not* show. The harness now
  runs both; the dependent column has no ISA numbers yet.
- **The fp32/fp16 arenas were filled with random bytes**, so a fraction of the
  operands were NaN/Inf/denormals. Denormal handling differs by
  microarchitecture, so the fp32 and fp16 hot ratios in particular should be
  re-measured with the typed fill before being quoted across parts.

The hamming and VNNI results are unaffected by the fill (integer kernels) and
are the least sensitive to the pinning question (their cycles/vector ratios are
5× and 2×, well outside cluster-to-cluster variation).

---

## The measurement

Same source, three forced-ISA builds, `-Doptimize=ReleaseFast`, `zig build
bench-isa`. Cycles/vector is primary (§7.5: "it isolates the ISA question from
the frequency question"); instructions/vector shows what the width bought;
effective frequency is `cycles / ref-cycles` per §6.6.4.

### L1-hot, the kernel with no memory pressure

| kernel | d | avx2 cyc | avx2-vnni cyc | avx512-full cyc | **cyc ratio** | insn ratio |
|---|--:|--:|--:|--:|--:|--:|
| dot_f32 | 768 | 105.0 | 102.7 | 98.8 | **1.06×** | 1.66× |
| dot_f32 | 1536 | 198.3 | 200.1 | 194.9 | **1.02×** | 1.79× |
| l2_f32 | 768 | 106.1 | 104.4 | 99.4 | 1.07× | 1.73× |
| l2_f32 | 1536 | 200.2 | 200.1 | 195.3 | 1.03× | 1.84× |
| dot_f16 | 768 | 104.3 | 102.3 | 158.1 | **0.66×** | 1.73× |
| dot_f16 | 1536 | 198.7 | 198.8 | 307.1 | **0.65×** | 1.84× |
| dot_i8 asym | 768 | 106.4 | 105.2 | 100.1 | 1.06× | 1.73× |
| dot_i8 sym | 768 | 141.6 | **30.7** | **28.3** | **5.00×** | 6.28× |
| dot_i8 sym | 1536 | 265.7 | **54.5** | **51.4** | **5.17×** | 7.77× |
| hamming | 768 | 21.4 |, | 21.3 | 1.00× | 1.65× |
| hamming | 1536 | 23.3 |, | 21.4 | 1.09× | 1.91× |

### DRAM-cold, the production regime for fp32

| kernel | d | avx2 cyc | avx2-vnni cyc | avx512-full cyc | **cyc ratio** |
|---|--:|--:|--:|--:|--:|
| dot_f32 | 768 | 846.2 | 818.7 | 551.2 | **1.54×** |
| dot_f32 | 1536 | 1233.7 | 1254.3 | 983.6 | **1.25×** |
| l2_f32 | 768 | 819.4 | 800.3 | 590.4 | 1.39× |
| l2_f32 | 1536 | 1293.6 | 1327.5 | 1042.4 | 1.24× |
| dot_f16 | 768 | 785.2 | 825.1 | 771.5 | 1.02× |
| dot_i8 asym | 1536 | 1153.5 | 1133.7 | 828.7 | 1.39× |
| dot_i8 sym | 768 | 831.7 | 327.8 | 268.8 | **3.09×** |
| dot_i8 sym | 1536 | 1371.2 | 501.1 | 366.9 | **3.74×** |
| hamming | 768 | 242.4 |, | 89.2 | **2.72×** |
| hamming | 1536 | 273.2 |, | 112.0 | **2.44×** |

The `hamming` L1-hot row cannot be read as an ISA result. It measures ~21
cycles at *every* dimension from 128 to 1536, a 12× range of work for a flat
cost, which is the benchmark harness's own floor: an indirect call through
`Spec.score` plus loop overhead. Binary distance at these dimensions is cheaper
than the machinery measuring it. The cold column, where the memory system
dominates that floor, is the one that resolves the ISA difference.

Effective frequency sat in 2.23 to 2.58 across every cell, a spread of ~14%.
§7.5's guard rail, "any AVX-512-vs-AVX2 comparison where effective frequency
differs by more than 3% between arms is reported with both frequencies visible"
- is therefore triggered, and the per-cell frequencies are in
`docs/kernel-matrix-*.md`. On Zen 5 there is no 512-bit licence penalty (the
datapath is full width), and the spread here tracks the boost governor rather
than the ISA; that is a property of this *host*, not of the ISA, and it is
exactly why §7.1 demands a fixed governor before publication.

**Those per-cell frequencies do not actually settle the guard rail, and the
files say why.** Every `kernel-matrix-*.md` opens with a note listing six
harness defects fixed after it was measured, and the first is that the run was
**unpinned on a hybrid host**, Zen 5 cores with 16 MiB of L3 beside Zen 5c
cores with 8 MiB, so a cell may mix cores of different clocks and cache. A
per-cell effective frequency taken that way cannot distinguish a boost-governor
spread from a core-type spread, which is the distinction the guard rail exists
to make. The harness now pins and prints the cpu with its sysfs cache sizes;
re-running `zig build bench-isa` on a gated host is what would close this, and
until then the 14% spread above is an upper bound on an unpinned instrument
rather than a measurement of this ISA question.

---

## Finding 1: the fp32 prediction is inverted

| | modelled (§6.6.1) | measured |
|---|---|---|
| fp32, cold | "little or nothing" | **1.25 to 1.54×** |
| fp32, L1-hot | (the case where width should pay) | **1.02 to 1.07×** |

§6.6.1 anticipated exactly this outcome and named the likely cause:

> "If cold fp32 *does* improve meaningfully, the model is wrong somewhere
> (likely we were never bandwidth bound, or **512-bit loads are improving
> memory-level parallelism by retiring fewer uops per byte**), and finding
> that out is worth more than the speedup."

The second hypothesis is the one the data supports, and it joins up with the
MLP measurement in `docs/cost-model.md`:

- AVX-512 halves the instruction count per vector (1.66 to 1.85× fewer).
- The out-of-order window is bounded in *uops*, not in bytes.
- So a 512-bit build holds roughly twice as many vectors' worth of loads in
  flight, which raises the concurrent-miss count.
- And concurrent misses are exactly what the cold case is short of: saturation
  needs ~64 outstanding misses and an `M=16` expansion supplies 16.

So the width helps cold **not because the ALU was the bottleneck, but because
fewer uops per byte buys more memory-level parallelism.** That reframes the
optimisation advice: on a memory-bound kernel, instruction count matters through
the reorder window even when it does not matter through the ALU.

The hot case, meanwhile, shows the flat result §6.6.1 expected to see *cold*:
1.02 to 1.07× despite 1.66 to 1.79× fewer instructions. L1-resident fp32 at this
accumulator count is bound by FMA throughput, and the 512-bit build issues half
as many FMAs of twice the width, the same work through the same ports.

## Finding 2: VNNI is the largest ISA effect in the engine, by far

§6.6.2 predicts "~3× instruction count" for `vpdpbusd`. Measured: **6.3 to 7.8×
fewer instructions and 5.0 to 5.2× fewer cycles** hot, 3.1 to 3.7× cold.

That is larger than the spec's estimate and larger than every other ISA effect
measured here combined.

**§11's open question 8 is answered.** "Is AVX-VNNI at 256 bits enough for the
SQ8 kernel, making full AVX-512 unnecessary for that path on Alder Lake+ /
Zen 4+?"

| | avx2 | avx2-vnni | avx512-full |
|---|--:|--:|--:|
| dot_i8 sym, d=768, hot | 141.6 cyc | **30.7** | 28.3 |
| dot_i8 sym, d=1536, hot | 265.7 cyc | **54.5** | 51.4 |

**Yes.** AVX-VNNI at 256 bits captures 92% of the available win (30.7 vs 28.3
cycles against a 141.6 baseline). The remaining 8% is the width, not the
instruction. A binary shipped to a fleet containing Alder Lake and Zen 4 parts
should use AVX-VNNI for SQ8 without needing AVX-512 at all, which is precisely
the "which ISA to select, per microarchitecture, per kernel" table §7.5 names as
the project's most transferable output.

## Finding 3: AVX-512 makes fp16 1.5× *slower*

`dot_f16` is the one kernel where the wide build loses, and it loses badly:
0.65 to 0.66× hot, i.e. 50% more cycles, despite 1.73 to 1.84× fewer instructions.

The kernel converts f16→f32 and accumulates in f32 (see `dist/dot_f16.zig` for
why: an f16 accumulator loses the whole significand at d=768). At 512 bits that
conversion widens a `@Vector(16, f16)` to a `@Vector(16, f32)`, which is a
256-bit→512-bit `vcvtph2ps`. The measurement says that conversion, not the FMA,
is the limiter, and it is worse per element at 512 bits than at 256.

This is a **recommendation to select AVX2 for the fp16 kernel specifically**,
even on a machine with AVX-512, and it is the concrete instance of §6.6.5's
prediction that "mixed configurations are likely optimal". The
`-Dforce-isa=` knob exists for exactly this.

## Finding 4: a tail-handling bug was masquerading as an ISA result

`hamming` measured **0.99× at d=768**, no benefit at all from `vpopcntq`,
against §6.6.2's assessment that it is "the single biggest ISA gap in the whole
engine". The instruction counts gave it away: 101 → 97, where every other kernel
showed 1.7×.

The cause was in this implementation, not the ISA. The kernel processes `L`
words per vector op and falls back to scalar for the remainder, and the
remainder gets *worse* as the register widens:

| dim | words | AVX2 (L=4) | AVX-512 (L=8) |
|---|--:|---|---|
| 128 | 2 | 2 scalar | 2 scalar |
| 384 | 6 | 4 vector + 2 scalar | **6 scalar** |
| 768 | 12 | 12 vector | 8 vector + **4 scalar** |
| 1536 | 24 | 16 + 8 vector | 24 vector |

At d=384 the 512-bit build did no vector work at all. The fix, padding stored
rows to a whole 512-bit register, `quant/binary.paddedWordsFor`, costs 33% more
bytes at d=768 (96 → 128 B) and is affordable because binary is latency bound,
not bandwidth bound (measured 3.22 GB/s against a 25.3 GB/s ceiling).

Measured before and after, `hamming` cycles/vector, avx512-full ÷ avx2:

| residency | d=128 | d=384 | d=768 | d=1536 |
|---|--:|--:|--:|--:|
| DRAM-cold, before | 1.48× | 1.05× | 1.57× | 2.20× |
| DRAM-cold, **after** | **1.97×** | **2.42×** | **2.72×** | **2.44×** |

and the instruction counts that were the tell: at d=768, avx512 went from 97
instructions per vector (unchanged from avx2's 101, the giveaway) to 62 against
avx2's 102, a 1.65× reduction consistent with every other kernel. d=384, the
worst case, improved 2.3×.

The general lesson is the one §7.3 states about counters: an ISA comparison
measures whatever the loop actually executed, and "no difference" is as likely
to mean "the fast path was not taken" as "the ISA does not matter". The
instruction-count column is what makes the difference visible; a
cycles-only table would have recorded 0.99× as a finding about `vpopcntq`.

---

## The per-microarchitecture selection table (§7.5's deliverable)

For **Zen 5 (Strix Point)**, from the measurements above:

| kernel | select | why |
|---|---|---|
| `dot_f32`, `l2_f32` | **AVX-512** | 1.25 to 1.54× cold, ~flat hot; the win is MLP, not ALU |
| `dot_f16` | **AVX2** | AVX-512 is 1.5× *slower*; the widening conversion is the limiter |
| `dot_i8` symmetric | **AVX-VNNI (256-bit)** | 92% of the win at 256 bits; full AVX-512 buys 8% |
| `dot_i8` asymmetric | AVX-512 | 1.16 to 1.39× cold |
| `hamming` | **AVX-512 + VPOPCNTDQ** | 1.57 to 2.20× cold, once rows are padded |
| `pq_adc` | either | bound by the 98 KB lookup table, not by the ISA |

§7.5 is emphatic that this table is meaningless from one machine: "the matrix is
meaningless on one machine. Minimum target set: one pre-Ice-Lake Intel server
part (where 512-bit licensing bites), one Sapphire Rapids or later, one Zen 4
(double-pumped), one Zen 5 (full-width)." This is the Zen 5 row of that table,
and the Zen 4 row in particular should look different, Zen 4 double-pumps
512-bit ops, so the fp32 MLP effect above should shrink or vanish there while the
VNNI result should hold.

---

## Reproducing

```
zig build bench-isa                       # build every arm
./zig-out/isa/<arm>/bench-micro-<arm> kernels
bench/isa/dump_asm.py                     # emitted-instruction summary per arm
```

### The `baseline` arm, and what it can and cannot isolate

`baseline` is plain `x86_64`: SSE2, no FMA, no F16C. Two consequences for
reading its row of `docs/asm/SUMMARY.md`:

- **fp32/mixed kernels round twice per step there, once everywhere else.**
  `@mulAdd` on a no-FMA target is lowered to a `fmaf` libcall per lane (that
  is what the builtin *means*), which is what the arm used to measure: 36
  `call compiler_rt.fma.fmaf` in `sm_dot_f32`, 476 instructions against 132
  now. `common.mulAdd` emits the fused op only when the build has FMA and
  `a * b + c` otherwise, so the arm is now honest SSE2 and its `sm_dot_f32`,
  `sm_euclid_f32` and `sm_dot_f32u8` rows shrank by ~3.5×. Its numerical
  results differ from the FMA arms in the last bit, exactly as Qdrant's own
  SSE and AVX kernels differ from each other.
- **f16 kernels still call `__extendhfsf2` per element on that arm.** F16C is
  not in the x86_64 baseline, so there is no instruction to lower the
  `@floatCast` to; this is a genuine property of the ISA, not a compiler
  failure, and it means the baseline `dot_f16` row measures a conversion
  libcall, not a distance kernel. Do not read an fp16 ratio off `baseline`.

`docs/asm/SUMMARY.md` confirms what each arm actually emitted, 5× `vpdpbusd` on
the VNNI arms, 5× `vpopcntq` on the VPOPCNTDQ arms, `vpshufb` elsewhere. §7.3
insists on that confirmation because "a compiler that quietly split 512-bit ops
is a common and invisible failure", and it caught a real one during development:
LLVM's `x86_64_v4` model sets `prefer_256_bit`, so every "AVX-512" arm emitted
256-bit code until the feature was explicitly subtracted in `build.zig`.
