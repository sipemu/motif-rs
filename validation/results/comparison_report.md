# motif-rs vs stumpy: Validation Comparison Report

Generated from `python validation/compare_results.py`

## Results Summary

| Test Case | Algorithm | MAD (profile) | Max Abs Diff | Correlation | Status |
|-----------|-----------|---------------|--------------|-------------|--------|
| sine_wave | stump | 2.73e-12 | 1.86e-11 | 1.00000000 | Excellent |
| square_wave | stump | 1.19e-11 | 1.10e-10 | 1.00000000 | Excellent |
| mixed_signal | stump | 3.19e-12 | 2.23e-11 | 1.00000000 | Excellent |
| streaming_sine | stumpi | 3.43e-14 | 9.43e-14 | 1.00000000 | Excellent |

## Performance Comparison

| Test Case | stumpy (s) | motif-rs (s) | Speedup |
|-----------|------------|--------------|---------|
| sine_wave | 14.287 | 0.701 | 20.4x |
| square_wave | 0.094 | 0.725 | 0.1x |
| mixed_signal | 0.090 | 0.687 | 0.1x |
| streaming_sine | 1.479 | 0.003 | 485.5x |

## Numerical Notes

- Floating point precision differences of ~4.2e-8 are expected for identical/linear subsequences due to IEEE 754 arithmetic in the distance formula
- Infinity sentinels (1e308) are used in JSON serialization since JSON does not support IEEE 754 infinity values
- MAD (Mean Absolute Difference) < 1e-6 with correlation > 0.999999 is classified as 'Excellent'

## Quality Tiers

| Tier | MAD Threshold | Correlation Threshold |
|------|---------------|-----------------------|
| Excellent | < 1e-6 | > 0.999999 |
| Good | < 1e-4 | > 0.9999 |
| Acceptable | < 1e-2 | > 0.99 |
| Concern | >= 1e-2 | <= 0.99 |
