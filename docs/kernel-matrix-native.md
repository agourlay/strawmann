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

isa build      : native
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
| dot_f32 | 128 | L1-hot | 4.6 | 23.8 | 82.0 | 3.45 | 2.572 | 110.18 |
| l2_f32 | 128 | L1-hot | 5.1 | 25.7 | 92.0 | 3.59 | 2.543 | 100.79 |
| dot_f16 | 128 | L1-hot | 7.2 | 34.7 | 90.0 | 2.60 | 2.441 | 35.66 |
| dot_i8_asym | 128 | L1-hot | 4.8 | 24.4 | 91.0 | 3.73 | 2.543 | 26.57 |
| dot_i8_sym | 128 | L1-hot | 4.2 | 21.2 | 65.0 | 3.07 | 2.535 | 30.47 |
| hamming | 128 | L1-hot | 4.2 | 21.0 | 55.0 | 2.61 | 2.541 | 15.38 |
| pq_adc_8 | 16 | L1-hot | 4.2 | 21.1 | 96.0 | 4.54 | 2.537 | 3.82 |
| dot_f32 | 128 | L3-resident | 5.9 | 29.4 | 82.0 | 2.79 | 2.516 | 86.97 |
| l2_f32 | 128 | L3-resident | 6.6 | 33.1 | 92.0 | 2.78 | 2.513 | 77.23 |
| dot_f16 | 128 | L3-resident | 25.3 | 58.7 | 90.0 | 1.53 | 1.173 | 10.11 |
| dot_i8_asym | 128 | L3-resident | 5.5 | 27.9 | 91.0 | 3.26 | 2.536 | 23.13 |
| dot_i8_sym | 128 | L3-resident | 4.5 | 22.5 | 65.0 | 2.88 | 2.518 | 28.48 |
| hamming | 128 | L3-resident | 4.3 | 21.5 | 55.0 | 2.55 | 2.516 | 14.89 |
| pq_adc_8 | 16 | L3-resident | 8.3 | 41.9 | 96.0 | 2.29 | 2.522 | 1.92 |
| dot_f32 | 128 | DRAM-cold | 41.3 | 201.6 | 82.0 | 0.41 | 2.460 | 12.40 |
| l2_f32 | 128 | DRAM-cold | 36.3 | 166.2 | 92.0 | 0.55 | 2.300 | 14.09 |
| dot_f16 | 128 | DRAM-cold | 41.6 | 198.9 | 90.0 | 0.45 | 2.397 | 6.15 |
| dot_i8_asym | 128 | DRAM-cold | 34.4 | 156.9 | 91.0 | 0.58 | 2.291 | 3.72 |
| dot_i8_sym | 128 | DRAM-cold | 22.1 | 98.3 | 65.0 | 0.66 | 2.233 | 5.79 |
| hamming | 128 | DRAM-cold | 26.3 | 120.6 | 55.0 | 0.46 | 2.306 | 2.44 |
| pq_adc_8 | 16 | DRAM-cold | 59.5 | 288.0 | 96.0 | 0.33 | 2.428 | 0.27 |
| dot_f32 | 384 | L1-hot | 10.2 | 51.7 | 124.0 | 2.40 | 2.538 | 150.46 |
| l2_f32 | 384 | L1-hot | 10.5 | 52.6 | 150.0 | 2.85 | 2.516 | 146.25 |
| dot_f16 | 384 | L1-hot | 18.1 | 87.4 | 148.0 | 1.69 | 2.433 | 42.53 |
| dot_i8_asym | 384 | L1-hot | 10.4 | 51.9 | 149.0 | 2.87 | 2.512 | 37.05 |
| dot_i8_sym | 384 | L1-hot | 4.4 | 22.1 | 83.0 | 3.75 | 2.516 | 86.97 |
| hamming | 384 | L1-hot | 4.2 | 21.0 | 55.0 | 2.61 | 2.512 | 15.23 |
| pq_adc_8 | 48 | L1-hot | 11.5 | 58.0 | 208.0 | 3.58 | 2.523 | 4.16 |
| dot_f32 | 384 | L3-resident | 20.4 | 95.1 | 124.0 | 1.30 | 2.349 | 75.27 |
| l2_f32 | 384 | L3-resident | 19.5 | 97.2 | 150.0 | 1.54 | 2.510 | 78.95 |
| dot_f16 | 384 | L3-resident | 18.3 | 91.5 | 148.0 | 1.62 | 2.506 | 41.89 |
| dot_i8_asym | 384 | L3-resident | 11.8 | 58.8 | 149.0 | 2.54 | 2.499 | 32.48 |
| dot_i8_sym | 384 | L3-resident | 5.6 | 27.7 | 83.0 | 3.00 | 2.495 | 68.96 |
| hamming | 384 | L3-resident | 5.7 | 28.5 | 55.0 | 1.93 | 2.497 | 11.19 |
| pq_adc_8 | 48 | L3-resident | 18.1 | 90.7 | 208.0 | 2.29 | 2.513 | 2.65 |
| dot_f32 | 384 | DRAM-cold | 80.8 | 404.5 | 124.0 | 0.31 | 2.520 | 19.00 |
| l2_f32 | 384 | DRAM-cold | 85.2 | 420.7 | 150.0 | 0.36 | 2.496 | 18.03 |
| dot_f16 | 384 | DRAM-cold | 90.7 | 457.1 | 148.0 | 0.32 | 2.533 | 8.47 |
| dot_i8_asym | 384 | DRAM-cold | 67.2 | 332.6 | 149.0 | 0.45 | 2.485 | 5.71 |
| dot_i8_sym | 384 | DRAM-cold | 28.8 | 145.3 | 83.0 | 0.57 | 2.539 | 13.32 |
| hamming | 384 | DRAM-cold | 18.1 | 76.9 | 55.0 | 0.72 | 2.140 | 3.53 |
| pq_adc_8 | 48 | DRAM-cold | 119.3 | 611.5 | 208.0 | 0.34 | 2.570 | 0.40 |
| dot_f32 | 768 | L1-hot | 19.7 | 99.2 | 187.0 | 1.89 | 2.528 | 155.97 |
| l2_f32 | 768 | L1-hot | 20.0 | 99.7 | 237.0 | 2.38 | 2.505 | 153.80 |
| dot_f16 | 768 | L1-hot | 32.0 | 160.4 | 235.0 | 1.46 | 2.514 | 47.98 |
| dot_i8_asym | 768 | L1-hot | 19.6 | 97.6 | 236.0 | 2.42 | 2.504 | 39.25 |
| dot_i8_sym | 768 | L1-hot | 5.8 | 28.9 | 103.0 | 3.57 | 2.505 | 132.87 |
| hamming | 768 | L1-hot | 4.2 | 21.1 | 62.0 | 2.94 | 2.492 | 30.16 |
| pq_adc_8 | 96 | L1-hot | 20.3 | 100.8 | 376.0 | 3.73 | 2.495 | 4.73 |
| dot_f32 | 768 | L3-resident | 31.2 | 155.9 | 187.0 | 1.20 | 2.504 | 98.36 |
| l2_f32 | 768 | L3-resident | 31.3 | 155.6 | 237.0 | 1.52 | 2.497 | 98.24 |
| dot_f16 | 768 | L3-resident | 34.1 | 169.8 | 235.0 | 1.38 | 2.498 | 45.01 |
| dot_i8_asym | 768 | L3-resident | 22.0 | 108.8 | 236.0 | 2.17 | 2.490 | 34.94 |
| dot_i8_sym | 768 | L3-resident | 10.1 | 50.1 | 103.0 | 2.06 | 2.482 | 75.81 |
| hamming | 768 | L3-resident | 4.8 | 23.6 | 62.0 | 2.63 | 2.470 | 26.68 |
| pq_adc_8 | 96 | L3-resident | 26.4 | 131.1 | 376.0 | 2.87 | 2.495 | 3.64 |
| dot_f32 | 768 | DRAM-cold | 114.4 | 569.7 | 187.0 | 0.33 | 2.507 | 26.85 |
| l2_f32 | 768 | DRAM-cold | 120.9 | 599.0 | 237.0 | 0.40 | 2.502 | 25.41 |
| dot_f16 | 768 | DRAM-cold | 148.8 | 744.8 | 235.0 | 0.32 | 2.514 | 10.33 |
| dot_i8_asym | 768 | DRAM-cold | 125.9 | 635.2 | 236.0 | 0.37 | 2.533 | 6.10 |
| dot_i8_sym | 768 | DRAM-cold | 57.7 | 274.5 | 103.0 | 0.38 | 2.393 | 13.32 |
| hamming | 768 | DRAM-cold | 20.3 | 90.8 | 62.0 | 0.68 | 2.245 | 6.29 |
| pq_adc_8 | 96 | DRAM-cold | 130.1 | 656.1 | 376.0 | 0.57 | 2.530 | 0.74 |
| dot_f32 | 1536 | L1-hot | 38.7 | 194.6 | 313.0 | 1.61 | 2.522 | 158.68 |
| l2_f32 | 1536 | L1-hot | 39.2 | 195.3 | 411.0 | 2.10 | 2.502 | 156.86 |
| dot_f16 | 1536 | L1-hot | 61.6 | 307.8 | 409.0 | 1.33 | 2.508 | 49.88 |
| dot_i8_asym | 1536 | L1-hot | 36.0 | 178.9 | 410.0 | 2.29 | 2.494 | 42.68 |
| dot_i8_sym | 1536 | L1-hot | 10.4 | 52.0 | 154.0 | 2.96 | 2.503 | 147.42 |
| hamming | 1536 | L1-hot | 4.3 | 21.4 | 69.0 | 3.22 | 2.494 | 44.54 |
| pq_adc_8 | 192 | L1-hot | 45.5 | 225.7 | 712.0 | 3.15 | 2.491 | 4.22 |
| dot_f32 | 1536 | L3-resident | 57.0 | 284.4 | 313.0 | 1.10 | 2.504 | 107.81 |
| l2_f32 | 1536 | L3-resident | 56.8 | 282.9 | 411.0 | 1.45 | 2.499 | 108.10 |
| dot_f16 | 1536 | L3-resident | 64.9 | 322.8 | 409.0 | 1.27 | 2.497 | 47.36 |
| dot_i8_asym | 1536 | L3-resident | 41.0 | 203.1 | 410.0 | 2.02 | 2.486 | 37.47 |
| dot_i8_sym | 1536 | L3-resident | 19.7 | 97.7 | 154.0 | 1.58 | 2.485 | 77.78 |
| hamming | 1536 | L3-resident | 4.9 | 24.0 | 69.0 | 2.87 | 2.463 | 39.24 |
| pq_adc_8 | 192 | L3-resident | 57.4 | 286.0 | 712.0 | 2.49 | 2.501 | 3.35 |
| dot_f32 | 1536 | DRAM-cold | 198.3 | 991.7 | 313.0 | 0.32 | 2.521 | 30.98 |
| l2_f32 | 1536 | DRAM-cold | 207.1 | 1041.7 | 411.0 | 0.39 | 2.533 | 29.67 |
| dot_f16 | 1536 | DRAM-cold | 191.2 | 957.0 | 409.0 | 0.43 | 2.517 | 16.07 |
| dot_i8_asym | 1536 | DRAM-cold | 163.6 | 817.0 | 410.0 | 0.50 | 2.516 | 9.39 |
| dot_i8_sym | 1536 | DRAM-cold | 82.2 | 401.8 | 154.0 | 0.38 | 2.456 | 18.68 |
| hamming | 1536 | DRAM-cold | 23.3 | 104.1 | 69.0 | 0.66 | 2.249 | 8.23 |
| pq_adc_8 | 192 | DRAM-cold | 254.3 | 1288.2 | 712.0 | 0.55 | 2.542 | 0.75 |

§6.6.1's prediction under test: the L1-hot and L3-resident rows
should track vector width; the DRAM-cold fp32 rows should not.
Compare this table across the arms built by `zig build bench-isa`.

