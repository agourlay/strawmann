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

isa build      : avx2
f32 lanes      : 8 (accumulators: 8)
dispatch tier  : avx2 (build max: avx2, host max: avx512)
vnni u8xi8     : false
perf counters  : available (effective frequency will be reported)
nominal TSC    : 1.996 GHz

## 2. Kernel matrix (§7.2.2, §7.5)

Cycles/vector is primary: it isolates the ISA question from the
frequency question. ns/vector is what actually matters and is
reported alongside, never instead (§7.5).

| kernel | d | residency | ns/vec | cyc/vec | insn/vec | IPC | eff.freq | GB/s |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| dot_f32 | 128 | L1-hot | 9.3 | 22.2 | 103.0 | 4.64 | 1.210 | 55.25 |
| l2_f32 | 128 | L1-hot | 6.6 | 22.9 | 121.0 | 5.29 | 1.739 | 77.23 |
| dot_f16 | 128 | L1-hot | 6.3 | 22.4 | 119.0 | 5.30 | 1.796 | 40.84 |
| dot_i8_asym | 128 | L1-hot | 6.7 | 24.0 | 119.0 | 4.96 | 1.797 | 19.07 |
| dot_i8_sym | 128 | L1-hot | 12.3 | 43.9 | 193.0 | 4.39 | 1.797 | 10.42 |
| hamming | 128 | L1-hot | 6.0 | 21.3 | 82.0 | 3.85 | 1.795 | 10.74 |
| pq_adc_8 | 16 | L1-hot | 5.9 | 21.0 | 98.0 | 4.66 | 1.797 | 2.72 |
| dot_f32 | 128 | L3-resident | 11.8 | 38.4 | 103.0 | 2.68 | 1.649 | 43.31 |
| l2_f32 | 128 | L3-resident | 12.5 | 44.6 | 121.0 | 2.71 | 1.797 | 41.00 |
| dot_f16 | 128 | L3-resident | 8.2 | 32.8 | 119.0 | 3.63 | 2.004 | 31.09 |
| dot_i8_asym | 128 | L3-resident | 6.5 | 33.5 | 119.0 | 3.55 | 2.574 | 19.57 |
| dot_i8_sym | 128 | L3-resident | 12.2 | 58.8 | 193.0 | 3.28 | 2.428 | 10.51 |
| hamming | 128 | L3-resident | 4.3 | 21.5 | 82.0 | 3.82 | 2.533 | 15.04 |
| pq_adc_8 | 16 | L3-resident | 7.9 | 39.9 | 98.0 | 2.46 | 2.536 | 2.02 |
| dot_f32 | 128 | DRAM-cold | 53.5 | 263.2 | 103.0 | 0.39 | 2.479 | 9.56 |
| l2_f32 | 128 | DRAM-cold | 59.3 | 303.4 | 121.0 | 0.40 | 2.576 | 8.63 |
| dot_f16 | 128 | DRAM-cold | 55.4 | 270.0 | 119.0 | 0.44 | 2.447 | 4.62 |
| dot_i8_asym | 128 | DRAM-cold | 54.8 | 265.6 | 119.0 | 0.45 | 2.440 | 2.34 |
| dot_i8_sym | 128 | DRAM-cold | 75.5 | 371.7 | 193.0 | 0.52 | 2.472 | 1.70 |
| hamming | 128 | DRAM-cold | 36.1 | 168.4 | 82.0 | 0.49 | 2.339 | 1.77 |
| pq_adc_8 | 16 | DRAM-cold | 70.3 | 352.9 | 98.0 | 0.28 | 2.518 | 0.23 |
| dot_f32 | 384 | L1-hot | 11.1 | 56.6 | 187.0 | 3.31 | 2.546 | 137.79 |
| l2_f32 | 384 | L1-hot | 11.1 | 55.8 | 237.0 | 4.24 | 2.513 | 137.79 |
| dot_f16 | 384 | L1-hot | 11.2 | 55.9 | 235.0 | 4.20 | 2.510 | 68.68 |
| dot_i8_asym | 384 | L1-hot | 11.3 | 56.5 | 235.0 | 4.16 | 2.509 | 33.97 |
| dot_i8_sym | 384 | L1-hot | 16.6 | 83.0 | 377.0 | 4.54 | 2.508 | 23.12 |
| hamming | 384 | L1-hot | 4.2 | 21.3 | 82.0 | 3.85 | 2.513 | 15.06 |
| pq_adc_8 | 48 | L1-hot | 11.7 | 58.8 | 210.0 | 3.57 | 2.525 | 4.11 |
| dot_f32 | 384 | L3-resident | 19.6 | 98.0 | 187.0 | 1.91 | 2.515 | 78.56 |
| l2_f32 | 384 | L3-resident | 19.2 | 95.8 | 237.0 | 2.47 | 2.508 | 80.13 |
| dot_f16 | 384 | L3-resident | 16.7 | 83.5 | 235.0 | 2.82 | 2.507 | 45.87 |
| dot_i8_asym | 384 | L3-resident | 16.8 | 83.9 | 235.0 | 2.80 | 2.509 | 22.89 |
| dot_i8_sym | 384 | L3-resident | 22.1 | 110.6 | 377.0 | 3.41 | 2.511 | 17.35 |
| hamming | 384 | L3-resident | 4.6 | 23.0 | 82.0 | 3.57 | 2.502 | 13.88 |
| pq_adc_8 | 48 | L3-resident | 18.3 | 91.5 | 210.0 | 2.29 | 2.518 | 2.63 |
| dot_f32 | 384 | DRAM-cold | 105.5 | 526.7 | 187.0 | 0.36 | 2.519 | 14.56 |
| l2_f32 | 384 | DRAM-cold | 108.9 | 545.4 | 237.0 | 0.43 | 2.520 | 14.11 |
| dot_f16 | 384 | DRAM-cold | 100.0 | 498.0 | 235.0 | 0.47 | 2.503 | 7.68 |
| dot_i8_asym | 384 | DRAM-cold | 128.0 | 644.4 | 235.0 | 0.36 | 2.531 | 3.00 |
| dot_i8_sym | 384 | DRAM-cold | 140.3 | 708.0 | 377.0 | 0.53 | 2.533 | 2.74 |
| hamming | 384 | DRAM-cold | 38.1 | 181.8 | 82.0 | 0.45 | 2.403 | 1.68 |
| pq_adc_8 | 48 | DRAM-cold | 134.7 | 677.4 | 210.0 | 0.31 | 2.524 | 0.36 |
| dot_f32 | 768 | L1-hot | 20.8 | 104.8 | 313.0 | 2.99 | 2.533 | 147.85 |
| l2_f32 | 768 | L1-hot | 21.1 | 105.3 | 411.0 | 3.90 | 2.500 | 145.39 |
| dot_f16 | 768 | L1-hot | 20.9 | 104.0 | 409.0 | 3.93 | 2.496 | 73.45 |
| dot_i8_asym | 768 | L1-hot | 20.9 | 104.2 | 409.0 | 3.93 | 2.495 | 36.66 |
| dot_i8_sym | 768 | L1-hot | 28.9 | 143.3 | 653.0 | 4.56 | 2.491 | 26.62 |
| hamming | 768 | L1-hot | 4.3 | 21.4 | 102.0 | 4.76 | 2.496 | 29.74 |
| pq_adc_8 | 96 | L1-hot | 20.4 | 101.4 | 378.0 | 3.73 | 2.493 | 4.71 |
| dot_f32 | 768 | L3-resident | 34.6 | 172.2 | 313.0 | 1.82 | 2.500 | 88.85 |
| l2_f32 | 768 | L3-resident | 35.5 | 176.8 | 411.0 | 2.33 | 2.498 | 86.47 |
| dot_f16 | 768 | L3-resident | 31.1 | 154.5 | 409.0 | 2.65 | 2.497 | 49.46 |
| dot_i8_asym | 768 | L3-resident | 30.0 | 149.7 | 409.0 | 2.73 | 2.502 | 25.56 |
| dot_i8_sym | 768 | L3-resident | 37.9 | 188.7 | 653.0 | 3.46 | 2.500 | 20.28 |
| hamming | 768 | L3-resident | 5.3 | 26.0 | 102.0 | 3.92 | 2.473 | 24.27 |
| pq_adc_8 | 96 | L3-resident | 27.3 | 136.3 | 378.0 | 2.77 | 2.506 | 3.52 |
| dot_f32 | 768 | DRAM-cold | 152.2 | 770.4 | 313.0 | 0.41 | 2.554 | 20.18 |
| l2_f32 | 768 | DRAM-cold | 169.6 | 848.8 | 411.0 | 0.48 | 2.513 | 18.11 |
| dot_f16 | 768 | DRAM-cold | 163.3 | 834.1 | 409.0 | 0.49 | 2.567 | 9.41 |
| dot_i8_asym | 768 | DRAM-cold | 155.7 | 781.4 | 409.0 | 0.52 | 2.528 | 4.93 |
| dot_i8_sym | 768 | DRAM-cold | 163.0 | 829.7 | 653.0 | 0.79 | 2.555 | 4.71 |
| hamming | 768 | DRAM-cold | 49.0 | 242.4 | 102.0 | 0.42 | 2.483 | 2.61 |
| pq_adc_8 | 96 | DRAM-cold | 141.9 | 718.2 | 378.0 | 0.53 | 2.541 | 0.68 |
| dot_f32 | 1536 | L1-hot | 40.0 | 201.1 | 565.0 | 2.81 | 2.519 | 153.41 |
| l2_f32 | 1536 | L1-hot | 40.3 | 200.2 | 759.0 | 3.79 | 2.493 | 152.52 |
| dot_f16 | 1536 | L1-hot | 40.2 | 199.8 | 757.0 | 3.79 | 2.493 | 76.39 |
| dot_i8_asym | 1536 | L1-hot | 40.3 | 200.4 | 757.0 | 3.78 | 2.493 | 38.09 |
| dot_i8_sym | 1536 | L1-hot | 54.0 | 268.0 | 1205.0 | 4.50 | 2.487 | 28.43 |
| hamming | 1536 | L1-hot | 4.7 | 23.3 | 132.0 | 5.67 | 2.495 | 41.03 |
| pq_adc_8 | 192 | L1-hot | 46.1 | 226.4 | 714.0 | 3.15 | 2.468 | 4.16 |
| dot_f32 | 1536 | L3-resident | 66.1 | 326.1 | 565.0 | 1.73 | 2.481 | 93.01 |
| l2_f32 | 1536 | L3-resident | 79.2 | 335.9 | 759.0 | 2.26 | 2.155 | 77.57 |
| dot_f16 | 1536 | L3-resident | 54.0 | 269.6 | 757.0 | 2.81 | 2.505 | 56.86 |
| dot_i8_asym | 1536 | L3-resident | 53.2 | 265.1 | 757.0 | 2.86 | 2.499 | 28.86 |
| dot_i8_sym | 1536 | L3-resident | 64.0 | 318.0 | 1205.0 | 3.79 | 2.494 | 23.99 |
| hamming | 1536 | L3-resident | 6.8 | 33.6 | 132.0 | 3.93 | 2.483 | 28.26 |
| pq_adc_8 | 192 | L3-resident | 56.3 | 281.0 | 714.0 | 2.54 | 2.506 | 3.41 |
| dot_f32 | 1536 | DRAM-cold | 234.4 | 1175.3 | 565.0 | 0.48 | 2.521 | 26.21 |
| l2_f32 | 1536 | DRAM-cold | 259.6 | 1311.9 | 759.0 | 0.58 | 2.541 | 23.66 |
| dot_f16 | 1536 | DRAM-cold | 225.7 | 1139.3 | 757.0 | 0.66 | 2.535 | 13.61 |
| dot_i8_asym | 1536 | DRAM-cold | 230.0 | 1162.0 | 757.0 | 0.65 | 2.535 | 6.68 |
| dot_i8_sym | 1536 | DRAM-cold | 267.5 | 1354.1 | 1205.0 | 0.89 | 2.540 | 5.74 |
| hamming | 1536 | DRAM-cold | 55.4 | 273.2 | 132.0 | 0.48 | 2.473 | 3.46 |
| pq_adc_8 | 192 | DRAM-cold | 265.4 | 1349.3 | 714.0 | 0.53 | 2.553 | 0.72 |

§6.6.1's prediction under test: the L1-hot and L3-resident rows
should track vector width; the DRAM-cold fp32 rows should not.
Compare this table across the arms built by `zig build bench-isa`.

