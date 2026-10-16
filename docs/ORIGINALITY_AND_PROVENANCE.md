# Originality and provenance note

This repository update is intended to make the empirical provenance of the CAL-TRANSFER manuscript explicit.

## External inputs

The empirical benchmark is EC-NAS / NAS-Bench-101-derived data. The benchmark itself is not claimed as an
original contribution of CAL-TRANSFER and should be cited to its original authors in both the paper and repository.

## Original analysis contributed here

The CAL-TRANSFER manuscript's original empirical analysis consists of the cross-GPU evaluation built on that
benchmark, including:

1. explicit separation of architecture ranking transfer from absolute joule-scale transfer;
2. a fixed-test repeated-split protocol that holds target-test composition unchanged while pilot policies vary;
3. matched-history global log-scale calibration with target pilot measurements;
4. deterministic coverage and quantile pilot-policy comparisons against expected random-pilot performance;
5. target measurement-energy accounting;
6. secondary local-residual, inductive robustness, and difficult-device sensitivity analyses.

## Textual originality

Repository documentation summarizes methods in newly written technical language. It does not reproduce
substantial passages from cited papers. External methods and benchmarks are identified by name and should
remain cited in the manuscript.

## Frozen outputs

`results/manuscript/` contains the numerical values reported in the final manuscript. These files document the
authors' analysis outputs; they do not represent copied tables from prior publications.
