# ConMedRL data processing

`ConMedRL.data` converts MIMIC-IV, SICdb, or an approved MIMIC-like dataset into the tables
expected by `TrainDataLoader` and `ValTestDataLoader`. CSV is always written;
Parquet, d3rlpy, and FHIR R4 exchange outputs are optional.

```python
from ConMedRL.data import build_dataset

bundle = build_dataset(
    database="sicdb",                 # or "mimic-iv"
    task="extubation",                # or "discharge"
    data_dir="/path/to/download",
    output_dir="./processed",
    output_formats=("csv", "parquet", "d3rlpy"),
    llm_provider="openai",            # optional
    llm_api_key=None,                 # or OPENAI_API_KEY
    min_variable_coverage=0.05,
)

print(bundle.summary())
display(bundle.schema.to_frame())
display(bundle.schema.dropped_frame())
```

In standard MIMIC-IV/SICdb runs, the LLM can only rank or translate dictionary
terms. The separate generic planner receives metadata only. Patient-level data
is never sent to an LLM provider. Verified source IDs and LOINC mappings are
used first, so an API key is not required for standard MIMIC-IV or SICdb.

The built-in extubation task minimises the extubation failure rate and has
exactly one constraint: `con_cost_0`, the remaining ICU length of stay in hours.
Reintubation/extubation-failure labels are used to construct the objective and
are not duplicated as a separate constraint cost.
See `Experiment Notebook/Example_ConMedRL_End_to_End_Workflow.ipynb` for the
complete preprocessing, ConMedRL loader, FQI/FQE, Lagrange update, held-out
evaluation, and optional interoperability workflow.

## Training with ConMedRL

```python
from ConMedRL import TrainDataLoader, ValTestDataLoader

train_loader = TrainDataLoader(
    cfg=rl_config,
    **bundle.loader_kwargs("train"),
)
train_loader.data_buffer_train(
    action_name=bundle.loader_action,
    done_condition=True,
    num_constraint=bundle.num_constraints,
)

val_loader = ValTestDataLoader(
    cfg=rl_config,
    **bundle.loader_kwargs("val"),
)
val_loader.data_buffer(
    action_name=bundle.loader_action,
    num_constraint=bundle.num_constraints,
)
```

## d3rlpy

ConMedRL minimises costs while d3rlpy maximises rewards. Conversion negates
costs by default:

```python
datasets = bundle.to_mdp_dataset("train")
objective_dataset = datasets.objective
first_constraint_dataset = datasets.constraints[0]
```

Discrete actions are integer encoded. Continuous actions keep the ordered
`bundle.action_columns`, float dtype, bounds, and vector shape. Configure the
continuous trainer with:

```python
from ConMedRL.conmedrl_continuous import RLTraining

training = RLTraining(
    rl_config,
    input_dim=bundle.state_dim,
    output_dim=bundle.action_dim,
    train_data_loader=train_loader.data_torch_loader_train,
    val_data_loader=val_loader.data_torch_loader,
    action_bounds=bundle.ordered_action_bounds,
)
```

## Unknown MIMIC-like datasets and LLM planning

The LLM drafts a data/task specification; it never writes or executes Python,
SQL, or expressions outside the safe operator whitelist. Only filenames,
headers, aggregate coverage, and recognized dictionary rows are sent remotely.
Patient rows remain local.

```python
import os
from ConMedRL.data import (
    GenericPlanner, PreprocessConfig, approve_plan, build_dataset,
    get_llm_backend, profile_dataset,
)
from ConMedRL.data.config import LLMConfig

profile = profile_dataset("/local/new_icu_dataset")
backend = get_llm_backend(LLMConfig(
    provider="openai",
    api_key=os.environ["OPENAI_API_KEY"],
))
draft = GenericPlanner(backend).recommend_task_plan(
    profile,
    "Hourly vasopressor dose optimization with mortality as objective "
    "and acute kidney injury as a safety constraint.",
)

# Review draft.to_dict(), warnings, evidence and unresolved decisions here.
approved = approve_plan(draft, profile)  # fails on stale/unsafe/unresolved plans
bundle = build_dataset(PreprocessConfig(
    database="generic",
    task=approved.tasks[0].name,
    data_dir="/local/new_icu_dataset",
    dataset_spec=approved,
))
```

Approval is content-addressed: changing a selected source file or any
executable choice invalidates the hash and blocks execution. Configuring an LLM
provider does not enable variable mapping during ordinary MIMIC/SICdb runs;
`llm_allow_mapping=True` is an explicit opt-in.

### Reviewed NWICU profile

NWICU is supplied as declarative JSON, not a site-specific Python adapter:

```python
from ConMedRL.data import ImputationConfig, PreprocessConfig, build_dataset
from ConMedRL.data.profiles import load_nwicu_profile

data_dir = "/path/to/nwicu"
dataset_spec, task_spec, approval_hash = load_nwicu_profile(
    "extubation", data_dir=data_dir
)
config = PreprocessConfig(
    database="generic",
    task="extubation",
    data_dir=data_dir,
    dataset_spec=dataset_spec,
    task_spec=task_spec,
    approved_plan_hash=approval_hash,
    imputation=ImputationConfig(knn_impute=False),
)
bundle = build_dataset(config)
```

The profile combines `chartevents` and `labevents`, aligns hospital labs to ICU
stays, converts Fahrenheit to Celsius and ounces to kilograms, and pairs
intubation/ventilation with later extubation point events. Concepts that cannot
be verified (including unplanned extubation) are reported as unavailable.

## FHIR R4 interoperability

FHIR conformance applies to exported exchange resources, not to scaled RL CSV
files, tensors, or models.

```python
bundle = build_dataset(
    database="mimic-iv",
    task="discharge",
    data_dir="/path/to/mimic",
    output_formats=("csv", "fhir"),
)
```

The exporter writes separate `Patient`, `Encounter`, `Observation`,
`Procedure`, `ConceptMap`, and `Provenance` NDJSON files plus a machine-readable
conformance report. IDs are pseudonymized; verified LOINC/UCUM mappings retain
the local coding in parallel. Offset-only sources use a relative-time
extension rather than fabricated dates. Set `FHIRConfig.validator_path` to the
official HL7 validator JAR for optional `-version 4.0` validation; otherwise
the report clearly says `structural-only`.

## Exact patient withdrawal and machine unlearning

ConMedRL supports auditable **exact retraining after patient withdrawal**. It
does not claim approximate weight scrubbing. Every stay and transition for a
requested `subject_id` is removed before observation extraction and caching;
the split, imputer, scaler, costs, FQI, and FQE must then be rebuilt.

```python
from ConMedRL import (
    ModelCompatibilityError,
    assert_model_compatible,
    exact_retrain,
)
from ConMedRL.data import rebuild_dataset_after_withdrawal

retained = rebuild_dataset_after_withdrawal(
    prior_manifest_path=bundle.written_files["manifest"],
    source_config=bundle.config,
    withdrawn_subject_ids=[10001234],
    output_dir="./processed_after_withdrawal",
    purge_prior_cache=True,
)

# A model manifest bound to the earlier dataset is rejected.
try:
    assert_model_compatible(old_model_manifest, retained)
except ModelCompatibilityError:
    pass

result = exact_retrain(
    retrain_callback=train_fresh_fqi_fqe,
    bundle=retained,
    seed=42,
    request_digest=retained.config.withdrawal_digest,
    invalidated_manifest=old_model_manifest,
)
assert result["method"] == "exact_retrain"
```

Raw withdrawal identifiers exist only in memory. Dataset manifests retain the
request count and SHA-256 digest, dataset content hash, parent manifest digest,
affected subject/episode/transition counts, and cache-purge audit. For a later
cumulative withdrawal, supply the complete withdrawn-ID set again because the
prior manifest is deliberately redacted. The superseded dataset/scaler
manifest is marked invalidated and `load_dataset` rejects it unless
`allow_invalidated=True` is explicitly used for audit. Model manifests bind FQI/FQE or
actor/critic artifacts to the retained dataset hash and reject invalidated,
modified, or stale artifacts. CUDA execution is seeded but is not promised to
be bitwise identical across different drivers and hardware.

See `Experiment Notebook/Example_Exact_Machine_Unlearning.ipynb` for a
self-contained synthetic baseline → withdrawal → stale-model rejection →
complete FQI/FQE retraining example.

## Leakage controls

- Train/validation/test assignment uses the task-configured grouping key so a
  complete decision trajectory is never split across partitions.
- Statistical outlier bounds, KNN imputation, and min-max scaling are fitted
  only on the training partition.
- Forward fill never crosses an ICU stay.
- Backward fill is disabled. Linear interpolation is opt-in because it uses a
  future measurement to fill an earlier policy state.
- State-space coverage is calculated from training data and unavailable or
  sparse variables are reported in the schema.

## SICdb details

- `Offset` is seconds since PDMS admission. The adapter subtracts
  `cases.ICUOffset` before converting to hours since ICU admission.
- Pre-ICU measurements are excluded.
- Extubation periods come from `data_range`:
  `DataID=720` (endotracheal tube), not MIMIC procedure item IDs.
- Urea (`Harnstoff`) is converted to BUN and magnesium is converted from
  mmol/L to mg/dL.
- SICdb has RASS but no GCS; the resolved state space therefore changes
  automatically.

## Reloading an exported dataset

```python
from ConMedRL.data import load_dataset

bundle = load_dataset("./processed/sicdb_extubation_manifest.json")
```
