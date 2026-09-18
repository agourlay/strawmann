> **Methodology note (added after the fact; numbers not re-measured).** This
> table was produced by the harness before the following fixes, so read it with
> them in mind: (1) the run was **unpinned** on a hybrid host (Zen 5 cpus
> 0-3,12-15 with 16 MiB L3; Zen 5c cpus 4-11,16-23 with 8 MiB L3), so cells may
> mix cores; the harness now pins and prints the cpu and its sysfs cache sizes.
> (2) The `L3-resident` and `DRAM-cold` rows are the **independent (MLP)** access
> loop, a throughput number with many misses in flight, not the dependent-fetch
> cost of spec §5.2; the harness now prints an `access` column and runs a
> `dependent` variant next to each. (3) `L3-resident` used a fixed 8 MiB working
> set, the whole L3 of a Zen 5c core; it is now 50% of the pinned core's L3.
> (4) Arenas were filled with random bytes, so the fp32/fp16 kernels saw
> NaN/Inf/denormal operands; they are now filled with finite values of the right
> type. (5) SQ8/PQ code rows were padded to 64 B; they are now at production
> stride (`dim` / `m` bytes). (6) The table now also carries a `huge%` column
> (fraction of the arena actually huge-backed, from smaps) and a per-row
> multiplexing warning when the perf group was not on the PMU for the whole
> window. See `docs/cost-model.md`, "Harness revision".

# strawmann microbenchmarks

isa build      : avx2-vnni
f32 lanes      : 8 (accumulators: 8)
dispatch tier  : avx2 (build max: avx2, host max: avx512)
vnni u8xi8     : true
perf counters  : available (effective frequency will be reported)
nominal TSC    : 1.996 GHz

## 2. Kernel matrix (§7.2.2, §7.5)

Cycles/vector is primary: it isolates the ISA question from the
frequency question. ns/vector is what actually matters and is
reported alongside, never instead (§7.5).

| kernel | d | residency | ns/vec | cyc/vec | insn/vec | IPC | eff.freq | GB/s |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| dot_f32 | 128 | L1-hot | 4.4 | 22.1 | 103.0 | 4.65 | 2.533 | 116.82 |
| l2_f32 | 128 | L1-hot | 4.7 | 23.5 | 121.0 | 5.15 | 2.505 | 108.91 |
| dot_f16 | 128 | L1-hot | 4.6 | 22.8 | 119.0 | 5.21 | 2.495 | 55.77 |
| dot_i8_asym | 128 | L1-hot | 4.8 | 23.9 | 119.0 | 4.98 | 2.493 | 26.60 |
| dot_i8_sym | 128 | L1-hot | 4.2 | 21.0 | 68.0 | 3.23 | 2.502 | 30.29 |
| hamming | 128 | L1-hot | 4.2 | 21.0 | 75.0 | 3.57 | 2.514 | 3.81 |
| pq_adc_8 | 16 | L1-hot | 4.2 | 21.0 | 98.0 | 4.66 | 2.504 | 3.80 |
| dot_f32 | 128 | L3-resident | 7.6 | 37.7 | 103.0 | 2.73 | 2.495 | 67.54 |
| l2_f32 | 128 | L3-resident | 9.1 | 44.9 | 121.0 | 2.69 | 2.494 | 56.54 |
| dot_f16 | 128 | L3-resident | 6.8 | 33.6 | 119.0 | 3.54 | 2.490 | 37.83 |
| dot_i8_asym | 128 | L3-resident | 6.9 | 34.1 | 119.0 | 3.48 | 2.490 | 18.60 |
| dot_i8_sym | 128 | L3-resident | 4.6 | 22.8 | 68.0 | 2.99 | 2.488 | 27.90 |
| hamming | 128 | L3-resident | 4.3 | 21.3 | 75.0 | 3.52 | 2.494 | 3.73 |
| pq_adc_8 | 16 | L3-resident | 8.1 | 40.3 | 98.0 | 2.43 | 2.497 | 1.98 |
| dot_f32 | 128 | DRAM-cold | 53.9 | 258.3 | 103.0 | 0.40 | 2.410 | 9.49 |
| l2_f32 | 128 | DRAM-cold | 47.1 | 240.6 | 121.0 | 0.50 | 2.576 | 10.88 |
| dot_f16 | 128 | DRAM-cold | 55.0 | 277.1 | 119.0 | 0.43 | 2.530 | 4.65 |
| dot_i8_asym | 128 | DRAM-cold | 53.8 | 275.5 | 119.0 | 0.43 | 2.576 | 2.38 |
| dot_i8_sym | 128 | DRAM-cold | 31.1 | 159.5 | 68.0 | 0.43 | 2.576 | 4.12 |
| hamming | 128 | DRAM-cold | 31.8 | 152.6 | 75.0 | 0.49 | 2.409 | 0.50 |
| pq_adc_8 | 16 | DRAM-cold | 68.2 | 340.2 | 98.0 | 0.29 | 2.502 | 0.23 |
| dot_f32 | 384 | L1-hot | 11.1 | 56.0 | 187.0 | 3.34 | 2.523 | 137.96 |
| l2_f32 | 384 | L1-hot | 11.2 | 55.3 | 237.0 | 4.29 | 2.493 | 137.71 |
| dot_f16 | 384 | L1-hot | 11.1 | 54.9 | 235.0 | 4.28 | 2.488 | 69.25 |
| dot_i8_asym | 384 | L1-hot | 11.4 | 56.3 | 235.0 | 4.17 | 2.488 | 33.80 |
| dot_i8_sym | 384 | L1-hot | 4.3 | 21.2 | 102.0 | 4.80 | 2.493 | 89.59 |
| hamming | 384 | L1-hot | 4.3 | 21.1 | 92.0 | 4.36 | 2.488 | 11.27 |
| pq_adc_8 | 48 | L1-hot | 11.7 | 58.4 | 210.0 | 3.59 | 2.502 | 4.09 |
| dot_f32 | 384 | L3-resident | 19.9 | 98.9 | 187.0 | 1.89 | 2.493 | 77.12 |
| l2_f32 | 384 | L3-resident | 19.2 | 95.0 | 237.0 | 2.50 | 2.488 | 80.20 |
| dot_f16 | 384 | L3-resident | 16.7 | 83.0 | 235.0 | 2.83 | 2.489 | 45.87 |
| dot_i8_asym | 384 | L3-resident | 16.6 | 82.4 | 235.0 | 2.85 | 2.491 | 23.15 |
| dot_i8_sym | 384 | L3-resident | 6.4 | 31.8 | 102.0 | 3.20 | 2.485 | 59.77 |
| hamming | 384 | L3-resident | 4.3 | 21.4 | 92.0 | 4.29 | 2.484 | 11.09 |
| pq_adc_8 | 48 | L3-resident | 18.8 | 93.5 | 210.0 | 2.25 | 2.500 | 2.55 |
| dot_f32 | 384 | DRAM-cold | 106.9 | 522.9 | 187.0 | 0.36 | 2.479 | 14.37 |
| l2_f32 | 384 | DRAM-cold | 107.5 | 532.0 | 237.0 | 0.45 | 2.504 | 14.29 |
| dot_f16 | 384 | DRAM-cold | 102.2 | 511.6 | 235.0 | 0.46 | 2.516 | 7.52 |
| dot_i8_asym | 384 | DRAM-cold | 120.8 | 620.2 | 235.0 | 0.38 | 2.576 | 3.18 |
| dot_i8_sym | 384 | DRAM-cold | 48.5 | 234.1 | 102.0 | 0.44 | 2.432 | 7.92 |
| hamming | 384 | DRAM-cold | 35.2 | 165.1 | 92.0 | 0.56 | 2.352 | 1.36 |
| pq_adc_8 | 48 | DRAM-cold | 127.5 | 638.0 | 210.0 | 0.33 | 2.511 | 0.38 |
| dot_f32 | 768 | L1-hot | 20.5 | 102.7 | 313.0 | 3.05 | 2.517 | 150.09 |
| l2_f32 | 768 | L1-hot | 21.0 | 104.4 | 411.0 | 3.94 | 2.489 | 146.09 |
| dot_f16 | 768 | L1-hot | 20.6 | 102.3 | 409.0 | 4.00 | 2.486 | 74.41 |
| dot_i8_asym | 768 | L1-hot | 21.2 | 105.2 | 409.0 | 3.89 | 2.489 | 36.22 |
| dot_i8_sym | 768 | L1-hot | 6.2 | 30.7 | 153.0 | 4.99 | 2.492 | 124.32 |
| hamming | 768 | L1-hot | 4.3 | 21.3 | 101.0 | 4.74 | 2.483 | 22.29 |
| pq_adc_8 | 96 | L1-hot | 20.5 | 101.5 | 378.0 | 3.73 | 2.482 | 4.68 |
| dot_f32 | 768 | L3-resident | 34.4 | 170.8 | 313.0 | 1.83 | 2.494 | 89.41 |
| l2_f32 | 768 | L3-resident | 35.2 | 174.4 | 411.0 | 2.36 | 2.489 | 87.39 |
| dot_f16 | 768 | L3-resident | 30.5 | 151.1 | 409.0 | 2.71 | 2.489 | 50.43 |
| dot_i8_asym | 768 | L3-resident | 29.9 | 148.1 | 409.0 | 2.76 | 2.492 | 25.73 |
| dot_i8_sym | 768 | L3-resident | 10.7 | 52.9 | 153.0 | 2.89 | 2.475 | 71.67 |
| hamming | 768 | L3-resident | 5.1 | 25.4 | 101.0 | 3.98 | 2.475 | 18.67 |
| pq_adc_8 | 96 | L3-resident | 27.1 | 134.4 | 378.0 | 2.81 | 2.492 | 3.55 |
| dot_f32 | 768 | DRAM-cold | 164.9 | 818.7 | 313.0 | 0.38 | 2.494 | 18.63 |
| l2_f32 | 768 | DRAM-cold | 157.9 | 800.3 | 411.0 | 0.51 | 2.550 | 19.45 |
| dot_f16 | 768 | DRAM-cold | 162.4 | 825.1 | 409.0 | 0.50 | 2.553 | 9.46 |
| dot_i8_asym | 768 | DRAM-cold | 155.8 | 789.7 | 409.0 | 0.52 | 2.545 | 4.93 |
| dot_i8_sym | 768 | DRAM-cold | 67.0 | 327.8 | 153.0 | 0.47 | 2.466 | 11.45 |
| hamming | 768 | DRAM-cold | 43.7 | 214.0 | 101.0 | 0.47 | 2.456 | 2.20 |
| pq_adc_8 | 96 | DRAM-cold | 140.4 | 709.7 | 378.0 | 0.53 | 2.536 | 0.68 |
| dot_f32 | 1536 | L1-hot | 40.0 | 200.1 | 565.0 | 2.82 | 2.508 | 153.46 |
| l2_f32 | 1536 | L1-hot | 40.5 | 200.1 | 759.0 | 3.79 | 2.481 | 151.79 |
| dot_f16 | 1536 | L1-hot | 40.2 | 198.8 | 757.0 | 3.81 | 2.480 | 76.38 |
| dot_i8_asym | 1536 | L1-hot | 40.4 | 199.8 | 757.0 | 3.79 | 2.481 | 38.03 |
| dot_i8_sym | 1536 | L1-hot | 11.0 | 54.5 | 255.0 | 4.68 | 2.487 | 139.69 |
| hamming | 1536 | L1-hot | 4.5 | 22.8 | 136.0 | 5.96 | 2.525 | 42.33 |
| pq_adc_8 | 192 | L1-hot | 47.0 | 227.0 | 714.0 | 3.14 | 2.433 | 4.08 |
| dot_f32 | 1536 | L3-resident | 65.8 | 326.8 | 565.0 | 1.73 | 2.495 | 93.41 |
| l2_f32 | 1536 | L3-resident | 75.3 | 337.1 | 759.0 | 2.25 | 2.264 | 81.57 |
| dot_f16 | 1536 | L3-resident | 54.8 | 272.0 | 757.0 | 2.78 | 2.491 | 56.04 |
| dot_i8_asym | 1536 | L3-resident | 54.1 | 267.5 | 757.0 | 2.83 | 2.484 | 28.41 |
| dot_i8_sym | 1536 | L3-resident | 20.6 | 100.2 | 255.0 | 2.55 | 2.453 | 74.70 |
| hamming | 1536 | L3-resident | 7.6 | 37.1 | 136.0 | 3.67 | 2.441 | 25.20 |
| pq_adc_8 | 192 | L3-resident | 57.9 | 286.7 | 714.0 | 2.49 | 2.487 | 3.32 |
| dot_f32 | 1536 | DRAM-cold | 249.4 | 1254.3 | 565.0 | 0.45 | 2.538 | 24.64 |
| l2_f32 | 1536 | DRAM-cold | 262.0 | 1327.5 | 759.0 | 0.57 | 2.544 | 23.45 |
| dot_f16 | 1536 | DRAM-cold | 218.9 | 1105.1 | 757.0 | 0.69 | 2.536 | 14.03 |
| dot_i8_asym | 1536 | DRAM-cold | 223.3 | 1133.7 | 757.0 | 0.67 | 2.549 | 6.88 |
| dot_i8_sym | 1536 | DRAM-cold | 101.5 | 501.1 | 255.0 | 0.51 | 2.485 | 15.13 |
| hamming | 1536 | DRAM-cold | 58.6 | 285.4 | 136.0 | 0.48 | 2.444 | 3.28 |
| pq_adc_8 | 192 | DRAM-cold | 267.1 | 1355.7 | 714.0 | 0.53 | 2.547 | 0.72 |

§6.6.1's prediction under test: the L1-hot and L3-resident rows
should track vector width; the DRAM-cold fp32 rows should not.
Compare this table across the arms built by `zig build bench-isa`.

