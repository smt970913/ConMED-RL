# Changelog

## 1.1.0 - 2026-09-08

### Added

- Unified `ConMedRL.data.build_dataset` processing for MIMIC-IV and SICdb.
- Reviewed declarative profiles for MIMIC-like datasets, including NWICU.
- `RLDatasetBundle` outputs for ConMED-RL loaders, CSV, Parquet, and d3rlpy.
- Pseudonymized FHIR R4 NDJSON export.
- Metadata-only LLM-assisted terminology mapping and processing-plan review.
- Patient-withdrawal reconstruction, content hashes, model compatibility
  checks, and caller-controlled fresh retraining.
- Bounded scalar and multidimensional continuous-action support.

### Changed

- Extubation uses extubation failure as the objective and unscaled remaining
  ICU length of stay as its single constraint.
- FQI/FQE training, target updates, multiplier projection, and deterministic
  seeding were corrected and tested.
- Packaging metadata, documentation, Docker examples, and experiment notebooks
  were updated for the unified 1.1.0 workflow.

### Compatibility

- The top-level `Data` package remains available for earlier code. New code
  should use `ConMedRL.data`.
