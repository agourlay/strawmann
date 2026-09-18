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

isa build      : avx512-full
f32 lanes      : 16 (accumulators: 8)
dispatch tier  : avx512 (build max: avx512, host max: avx512)
vnni u8xi8     : true
perf counters  : available (effective frequency will be reported)
nominal TSC    : 1.996 GHz

## 2. Kernel matrix (§7.2.2, §7.5)

Cycles/vector is primary: it isolates the ISA question from the
frequency question. ns/vector is what actually matters and is
reported alongside, never instead (§7.5).

| kernel | d | residency | ns/vec | cyc/vec | insn/vec | IPC | eff.freq | GB/s |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| dot_f32 | 128 | L1-hot | 4.8 | 24.0 | 84.0 | 3.49 | 2.534 | 107.33 |
| l2_f32 | 128 | L1-hot | 5.2 | 25.9 | 93.0 | 3.60 | 2.510 | 98.92 |
| dot_f16 | 128 | L1-hot | 14.1 | 33.5 | 92.0 | 2.74 | 1.201 | 18.20 |
| dot_i8_asym | 128 | L1-hot | 9.0 | 24.8 | 91.0 | 3.67 | 1.413 | 14.20 |
| dot_i8_sym | 128 | L1-hot | 4.4 | 22.2 | 65.0 | 2.92 | 2.558 | 29.34 |
| hamming | 128 | L1-hot | 4.6 | 21.1 | 55.0 | 2.61 | 2.289 | 13.78 |
| pq_adc_8 | 16 | L1-hot | 4.2 | 21.0 | 98.0 | 4.66 | 2.523 | 3.82 |
| dot_f32 | 128 | L3-resident | 5.8 | 29.0 | 84.0 | 2.89 | 2.502 | 87.71 |
| l2_f32 | 128 | L3-resident | 6.4 | 32.1 | 93.0 | 2.90 | 2.515 | 79.85 |
| dot_f16 | 128 | L3-resident | 7.0 | 35.0 | 92.0 | 2.63 | 2.503 | 36.46 |
| dot_i8_asym | 128 | L3-resident | 6.5 | 32.4 | 91.0 | 2.81 | 2.499 | 19.66 |
| dot_i8_sym | 128 | L3-resident | 4.6 | 22.9 | 65.0 | 2.84 | 2.496 | 27.76 |
| hamming | 128 | L3-resident | 4.4 | 21.8 | 55.0 | 2.52 | 2.496 | 14.57 |
| pq_adc_8 | 16 | L3-resident | 9.3 | 46.1 | 98.0 | 2.13 | 2.496 | 1.73 |
| dot_f32 | 128 | DRAM-cold | 30.4 | 137.2 | 84.0 | 0.61 | 2.267 | 16.84 |
| l2_f32 | 128 | DRAM-cold | 37.3 | 179.8 | 93.0 | 0.52 | 2.424 | 13.71 |
| dot_f16 | 128 | DRAM-cold | 43.4 | 203.5 | 92.0 | 0.45 | 2.361 | 5.90 |
| dot_i8_asym | 128 | DRAM-cold | 34.4 | 161.9 | 91.0 | 0.56 | 2.365 | 3.72 |
| dot_i8_sym | 128 | DRAM-cold | 19.5 | 89.4 | 65.0 | 0.73 | 2.305 | 6.57 |
| hamming | 128 | DRAM-cold | 19.0 | 85.4 | 55.0 | 0.64 | 2.264 | 3.37 |
| pq_adc_8 | 16 | DRAM-cold | 59.2 | 286.7 | 98.0 | 0.34 | 2.431 | 0.27 |
| dot_f32 | 384 | L1-hot | 10.2 | 51.1 | 126.0 | 2.47 | 2.517 | 150.34 |
| l2_f32 | 384 | L1-hot | 10.6 | 52.9 | 151.0 | 2.85 | 2.499 | 144.33 |
| dot_f16 | 384 | L1-hot | 17.3 | 86.4 | 150.0 | 1.74 | 2.505 | 44.31 |
| dot_i8_asym | 384 | L1-hot | 11.1 | 55.3 | 149.0 | 2.70 | 2.498 | 34.54 |
| dot_i8_sym | 384 | L1-hot | 4.9 | 24.3 | 84.0 | 3.46 | 2.505 | 78.78 |
| hamming | 384 | L1-hot | 4.2 | 21.1 | 55.0 | 2.61 | 2.498 | 15.12 |
| pq_adc_8 | 48 | L1-hot | 11.8 | 58.8 | 210.0 | 3.57 | 2.510 | 4.08 |
| dot_f32 | 384 | L3-resident | 19.7 | 98.0 | 126.0 | 1.29 | 2.501 | 77.92 |
| l2_f32 | 384 | L3-resident | 20.1 | 99.7 | 151.0 | 1.51 | 2.494 | 76.41 |
| dot_f16 | 384 | L3-resident | 18.6 | 92.1 | 150.0 | 1.63 | 2.487 | 41.26 |
| dot_i8_asym | 384 | L3-resident | 12.6 | 62.2 | 149.0 | 2.40 | 2.481 | 30.48 |
| dot_i8_sym | 384 | L3-resident | 5.5 | 27.0 | 84.0 | 3.11 | 2.476 | 70.17 |
| hamming | 384 | L3-resident | 4.3 | 21.4 | 55.0 | 2.56 | 2.480 | 14.74 |
| pq_adc_8 | 48 | L3-resident | 18.3 | 91.3 | 210.0 | 2.30 | 2.498 | 2.62 |
| dot_f32 | 384 | DRAM-cold | 79.3 | 384.5 | 126.0 | 0.33 | 2.440 | 19.37 |
| l2_f32 | 384 | DRAM-cold | 75.2 | 365.2 | 151.0 | 0.41 | 2.444 | 20.41 |
| dot_f16 | 384 | DRAM-cold | 89.3 | 437.1 | 150.0 | 0.34 | 2.463 | 8.60 |
| dot_i8_asym | 384 | DRAM-cold | 67.9 | 330.5 | 149.0 | 0.45 | 2.450 | 5.66 |
| dot_i8_sym | 384 | DRAM-cold | 29.4 | 132.4 | 84.0 | 0.63 | 2.267 | 13.05 |
| hamming | 384 | DRAM-cold | 18.2 | 75.0 | 55.0 | 0.73 | 2.068 | 3.52 |
| pq_adc_8 | 48 | DRAM-cold | 119.3 | 609.0 | 210.0 | 0.34 | 2.561 | 0.40 |
| dot_f32 | 768 | L1-hot | 19.6 | 98.7 | 189.0 | 1.91 | 2.522 | 156.45 |
| l2_f32 | 768 | L1-hot | 19.9 | 99.2 | 238.0 | 2.40 | 2.501 | 154.38 |
| dot_f16 | 768 | L1-hot | 31.6 | 158.1 | 237.0 | 1.50 | 2.507 | 48.54 |
| dot_i8_asym | 768 | L1-hot | 20.1 | 100.2 | 236.0 | 2.36 | 2.500 | 38.21 |
| dot_i8_sym | 768 | L1-hot | 5.6 | 28.2 | 104.0 | 3.69 | 2.504 | 135.94 |
| hamming | 768 | L1-hot | 4.3 | 21.3 | 62.0 | 2.91 | 2.497 | 29.94 |
| pq_adc_8 | 96 | L1-hot | 20.4 | 101.4 | 378.0 | 3.73 | 2.491 | 4.70 |
| dot_f32 | 768 | L3-resident | 37.4 | 167.2 | 189.0 | 1.13 | 2.257 | 82.09 |
| l2_f32 | 768 | L3-resident | 30.7 | 154.1 | 238.0 | 1.54 | 2.518 | 99.95 |
| dot_f16 | 768 | L3-resident | 33.5 | 166.5 | 237.0 | 1.42 | 2.496 | 45.86 |
| dot_i8_asym | 768 | L3-resident | 23.3 | 110.2 | 236.0 | 2.14 | 2.376 | 32.93 |
| dot_i8_sym | 768 | L3-resident | 9.9 | 49.6 | 104.0 | 2.10 | 2.506 | 77.34 |
| hamming | 768 | L3-resident | 4.5 | 22.2 | 62.0 | 2.79 | 2.491 | 28.63 |
| pq_adc_8 | 96 | L3-resident | 26.7 | 133.0 | 378.0 | 2.84 | 2.502 | 3.59 |
| dot_f32 | 768 | DRAM-cold | 113.4 | 561.1 | 189.0 | 0.34 | 2.494 | 27.09 |
| l2_f32 | 768 | DRAM-cold | 119.3 | 592.8 | 238.0 | 0.40 | 2.504 | 25.75 |
| dot_f16 | 768 | DRAM-cold | 146.9 | 725.5 | 237.0 | 0.33 | 2.479 | 10.46 |
| dot_i8_asym | 768 | DRAM-cold | 127.5 | 638.9 | 236.0 | 0.37 | 2.516 | 6.02 |
| dot_i8_sym | 768 | DRAM-cold | 55.5 | 266.5 | 104.0 | 0.39 | 2.419 | 13.84 |
| hamming | 768 | DRAM-cold | 20.9 | 89.2 | 62.0 | 0.70 | 2.144 | 6.12 |
| pq_adc_8 | 96 | DRAM-cold | 132.8 | 666.1 | 378.0 | 0.57 | 2.517 | 0.72 |
| dot_f32 | 1536 | L1-hot | 38.8 | 194.9 | 315.0 | 1.62 | 2.520 | 158.39 |
| l2_f32 | 1536 | L1-hot | 39.2 | 195.3 | 412.0 | 2.11 | 2.497 | 156.55 |
| dot_f16 | 1536 | L1-hot | 61.3 | 305.7 | 411.0 | 1.34 | 2.501 | 50.09 |
| dot_i8_asym | 1536 | L1-hot | 36.9 | 182.4 | 410.0 | 2.25 | 2.482 | 41.64 |
| dot_i8_sym | 1536 | L1-hot | 10.3 | 51.3 | 155.0 | 3.02 | 2.501 | 149.20 |
| hamming | 1536 | L1-hot | 4.3 | 21.4 | 69.0 | 3.23 | 2.493 | 44.64 |
| pq_adc_8 | 192 | L1-hot | 46.1 | 226.2 | 714.0 | 3.16 | 2.463 | 4.16 |
| dot_f32 | 1536 | L3-resident | 57.0 | 284.8 | 315.0 | 1.11 | 2.507 | 107.79 |
| l2_f32 | 1536 | L3-resident | 57.3 | 284.0 | 412.0 | 1.45 | 2.488 | 107.21 |
| dot_f16 | 1536 | L3-resident | 65.1 | 322.7 | 411.0 | 1.27 | 2.489 | 47.21 |
| dot_i8_asym | 1536 | L3-resident | 41.2 | 203.8 | 410.0 | 2.01 | 2.483 | 37.29 |
| dot_i8_sym | 1536 | L3-resident | 18.9 | 93.7 | 155.0 | 1.65 | 2.487 | 81.21 |
| hamming | 1536 | L3-resident | 5.7 | 27.9 | 69.0 | 2.47 | 2.463 | 33.77 |
| pq_adc_8 | 192 | L3-resident | 56.6 | 281.6 | 714.0 | 2.54 | 2.495 | 3.39 |
| dot_f32 | 1536 | DRAM-cold | 200.7 | 1005.5 | 315.0 | 0.31 | 2.518 | 30.62 |
| l2_f32 | 1536 | DRAM-cold | 199.0 | 992.3 | 412.0 | 0.42 | 2.515 | 30.88 |
| dot_f16 | 1536 | DRAM-cold | 182.6 | 909.0 | 411.0 | 0.45 | 2.507 | 16.83 |
| dot_i8_asym | 1536 | DRAM-cold | 173.4 | 829.4 | 410.0 | 0.49 | 2.412 | 8.86 |
| dot_i8_sym | 1536 | DRAM-cold | 79.6 | 398.7 | 155.0 | 0.39 | 2.527 | 19.29 |
| hamming | 1536 | DRAM-cold | 23.8 | 112.0 | 69.0 | 0.62 | 2.367 | 8.05 |
| pq_adc_8 | 192 | DRAM-cold | 252.7 | 1283.1 | 714.0 | 0.56 | 2.547 | 0.76 |

§6.6.1's prediction under test: the L1-hot and L3-resident rows
should track vector width; the DRAM-cold fp32 rows should not.
Compare this table across the arms built by `zig build bench-isa`.

