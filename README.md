# ConMED-RL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/conmedrl.svg)](https://pypi.org/project/conmedrl/)

<p align="center">
  <img src="image/ConMED-RL Logo.png" width="400" alt="ConMED-RL logo">
</p>

ConMED-RL is an open-source Python toolkit for offline constrained
reinforcement learning (OCRL) in critical-care research. It connects:

1. processing of local ICU data into trajectory-aligned offline RL datasets;
2. discrete FQI/FQE and continuous actor-critic/FQE training;
3. separate objective and constraint evaluation;
4. optional interoperability, data-withdrawal, and research-interface tools.

The reference applications are ICU discharge and extubation decision-making.
The package is intended for retrospective research and software evaluation. It
is not a medical device and must not be used to direct patient care without
appropriate local validation, governance, regulatory review, and clinician
oversight.

The current PyPI and GitHub release is **1.1.0**.

## What is included in version 1.1.0

- A unified `ConMedRL.data.build_dataset` API for MIMIC-IV and SICdb.
- A reviewed declarative adapter for MIMIC-like sources, including NWICU
  example profiles.
- Dynamic state-space resolution based on variables available in the selected
  database.
- Task-specific, trajectory-preserving train/validation/test splitting.
- Training-only fitting of imputation, outlier handling, and scaling.
- `RLDatasetBundle`, the common contract between preprocessing, ConMED-RL
  loaders, and external offline RL libraries.
- CSV output plus optional Parquet and `d3rlpy.dataset.MDPDataset` exports.
- Optional pseudonymized FHIR R4 NDJSON export.
- Metadata-only LLM assistance for terminology search and reviewed processing
  plans. Patient rows are not sent to an LLM.
- Exact dataset reconstruction after a local patient-withdrawal request,
  invalidation of superseded manifests, model compatibility checks, and
  caller-controlled fresh retraining.
- Discrete and bounded multi-dimensional continuous action support.
- Deterministic seeding and dataset/model content hashes for reproducibility.

For implementation details and safety boundaries, see
[`DATA_PROCESSING.md`](DATA_PROCESSING.md).

## Installation

Install the released package:

```bash
pip install conmedrl
```

Install optional data interoperability or LLM dependencies:

```bash
pip install "conmedrl[data]"
pip install "conmedrl[llm]"
pip install "conmedrl[data,llm]"
```

For development and the example notebooks:

```bash
git clone https://github.com/smt970913/ConMED-RL.git
cd ConMED-RL
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[data,llm,dev]"
```

Python 3.10.14, as recorded in `runtime.txt`, is the recommended repository
environment. Core algorithms can run on CPU or CUDA.

## Unified data-processing API

Raw clinical data remain on the user's machine. The following call builds a
training-ready dataset and writes CSV files:

```python
from ConMedRL import build_dataset

bundle = build_dataset(
    database="mimic-iv",             # "mimic-iv" or "sicdb"
    task="discharge",                # "discharge" or "extubation"
    data_dir="/path/to/mimic-iv",
    output_dir="./processed",
    output_formats=("csv", "parquet", "d3rlpy"),
    llm_provider="none",
)

print(bundle.summary())
print(bundle.schema.names)
print(bundle.num_constraints)
```

The returned `RLDatasetBundle` contains aligned state/outcome tables for the
train, validation, and test splits; the terminal state; ordered state and
action metadata; objective and constraint costs; fitted transformations; and a
content hash.

For the built-in extubation task, the objective is extubation failure and the
single constraint is remaining ICU length of stay in hours. Reintubation
contributes to the extubation-failure definition and is not a separate
constraint. ICU length of stay is not scaled as a cost.

## Training with ConMED-RL

`RLDatasetBundle.loader_kwargs` supplies the tables expected by the existing
PyTorch-based loaders:

```python
from ConMedRL import TrainDataLoader, ValTestDataLoader

train_loader = TrainDataLoader(
    cfg=rl_config,
    **bundle.loader_kwargs("train"),
)
train_loader.data_buffer_train(
    action_name=bundle.loader_action,
    done_condition=None,
    num_constraint=bundle.num_constraints,
)

val_loader = ValTestDataLoader(
    cfg=rl_config,
    **bundle.loader_kwargs("val"),
)
val_loader.data_buffer(
    action_name=bundle.loader_action,
    done_condition=None,
    num_constraint=bundle.num_constraints,
)
```

The resolved dimensions can then configure `RLTraining`:

```python
from ConMedRL import RLTraining

trainer = RLTraining(
    cfg=rl_config,
    state_dim=bundle.state_dim,
    action_dim=bundle.action_dim,
    train_data_loader=train_loader.data_torch_loader_train,
    val_data_loader=val_loader.data_torch_loader,
)
```

ConMED-RL creates one FQE estimator for the objective and one for each
constraint. FQE is fitted on training transitions and evaluated on held-out
validation decision states during multiplier updates. The test split remains
unused during training and model selection.

See
[`Example_ConMedRL_End_to_End_Workflow.ipynb`](Experiment%20Notebook/Example_ConMedRL_End_to_End_Workflow.ipynb)
for a short complete workflow.

## d3rlpy interoperability

ConMED-RL minimizes costs, whereas `d3rlpy` algorithms maximize rewards.
Conversion therefore negates costs by default and keeps objective and
constraint datasets separate:

```python
datasets = bundle.to_mdp_dataset(
    split="train",
    negate_costs=True,
    include_constraints=True,
)

objective_dataset = datasets.objective
constraint_datasets = datasets.constraints
```

Install this integration with `pip install "conmedrl[data]"`.

## Generic MIMIC-like data and reviewed LLM planning

New MIMIC-like sources do not require a new Python adapter. `DatasetSpec` and
`TaskSpec` documents define local files, columns, time variables, unit
conversions, aggregation, actions, terminal events, objectives, and
constraints. Specifications are validated locally and require an approval hash
before execution.

An optional LLM can draft mappings from dataset metadata and variable
dictionaries. It cannot approve a plan, execute generated Python or SQL, or
receive patient rows. Clinical definitions and all proposed mappings require
human review.

The packaged NWICU profiles demonstrate this path:

```python
from ConMedRL.data import PreprocessConfig, build_dataset
from ConMedRL.data.profiles import load_nwicu_profile

data_dir = "/path/to/nwicu"
dataset_spec, task_spec, approval_hash = load_nwicu_profile(
    "extubation",
    data_dir=data_dir,
)

config = PreprocessConfig(
    database="generic",
    task="extubation",
    data_dir=data_dir,
    output_dir="./processed_nwicu",
    dataset_spec=dataset_spec,
    task_spec=task_spec,
    approved_plan_hash=approval_hash,
)

bundle = build_dataset(config)
```

Relevant examples:

- [`Example_Generic_MIMIC_Like_NWICU.ipynb`](Experiment%20Notebook/Example_Generic_MIMIC_Like_NWICU.ipynb)
- [`Example_LLM_Custom_Clinical_Task.ipynb`](Experiment%20Notebook/Example_LLM_Custom_Clinical_Task.ipynb)

## FHIR R4 export

FHIR applies to exchange resources, not to scaled RL tensors or model files.
Requesting the `fhir` output writes pseudonymized `Patient`, `Encounter`,
`Observation`, `Procedure`, `ConceptMap`, and `Provenance` NDJSON files:

```python
bundle = build_dataset(
    database="mimic-iv",
    task="discharge",
    data_dir="/path/to/mimic-iv",
    output_dir="./processed_fhir",
    output_formats=("csv", "fhir"),
)
```

The exporter performs structural checks. Formal conformance requires a
separately configured official HL7 validator. See
[`Example_FHIR_R4_Interop.ipynb`](Experiment%20Notebook/Example_FHIR_R4_Interop.ipynb).

## Patient withdrawal and fresh retraining

For locally held data, ConMED-RL can reconstruct a successor dataset after a
patient-withdrawal request:

```python
from ConMedRL import rebuild_dataset_after_withdrawal

retained = rebuild_dataset_after_withdrawal(
    prior_manifest_path=bundle.written_files["manifest"],
    source_config=bundle.config,
    withdrawn_subject_ids=[10001234],
    output_dir="./processed_after_withdrawal",
)
```

The operation removes all associated episodes before feature extraction,
rebuilds the splits and transformations, records a new content hash, and
invalidates the superseded dataset manifest. Model manifests bound to the
earlier hash are rejected. The separate `exact_retrain` helper invokes a
caller-provided callback that must create fresh model and optimizer state.

This is deletion followed by reconstruction and retraining, not approximate
parameter scrubbing or proof of erasure from external systems. See
[`Example_Exact_Machine_Unlearning.ipynb`](Experiment%20Notebook/Example_Exact_Machine_Unlearning.ipynb).

## Continuous actions

The continuous module supports bounded scalar or vector actions:

```python
from ConMedRL.conmedrl_continuous import RLTraining

trainer = RLTraining(
    cfg=rl_config,
    input_dim=bundle.state_dim,
    output_dim=bundle.action_dim,
    train_data_loader=train_loader.data_torch_loader_train,
    val_data_loader=val_loader.data_torch_loader,
    action_bounds=bundle.ordered_action_bounds,
)
```

This component is provided for methodological research. It has not been
validated for autonomous clinical control.

## Data sources

Raw clinical datasets are not distributed with this repository:

- [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
- [SICdb](https://physionet.org/content/sicdb/1.0.8/)
- [NWICU](https://physionet.org/content/nwicu-northwestern-icu/0.1.0/)

Users are responsible for obtaining access, satisfying data-use agreements,
and following institutional privacy and ethics requirements.

## Repository layout

```text
ConMedRL/
  conmedrl.py                 discrete FQI/FQE OCRL implementation
  conmedrl_continuous.py      continuous actor-critic/FQE implementation
  data_loader.py              PyTorch transition loaders
  model_artifacts.py          model manifests and compatibility checks
  data/
    adapters/                 MIMIC-IV, SICdb, and generic adapters
    profiles/                 reviewed NWICU specifications
    pipeline.py               unified build/load API
    dataset.py                RLDatasetBundle and d3rlpy conversion
    fhir.py                   FHIR R4 export
    planner.py                reviewed metadata-level planning
    unlearning.py             patient-withdrawal reconstruction
Experiment Notebook/         executable examples
tests/                       unit and integration tests
CDM-Software/                local Flask research demonstration
```

The top-level `Data` package is retained for backward compatibility. New code
should use `ConMedRL.data`.

## Research software interfaces

`CDM-Software/web_application_demo.py` is a local Flask demonstration for
checking trained-model integration across the discharge and extubation
examples. A focused extubation research prototype is available at
[ExtEval](https://exteval.com/).

These interfaces are research prototypes. Their outputs are not clinical
recommendations and do not establish clinical effectiveness or safety.

## Research background

The discharge formulation and evaluation are reported in:

> Sun, M. and Xie, J. (2025). *Discharge with Multiple Readmissions and
> Constraints*. IISE Transactions on Healthcare Systems Engineering.
> https://doi.org/10.1080/24725579.2025.2569355

The extubation study, *Personalized Extubation Decisions under Resource
Constraints: An Offline Constrained Reinforcement Learning Approach*, is under
major revision at *Health Care Management Science*.

## License and contact

ConMED-RL is released under the [MIT License](LICENSE).

- Maotong Sun: maotong.sun@tum.de
- Jingui Xie: jingui.xie@tum.de

Please use [GitHub Issues](https://github.com/smt970913/ConMED-RL/issues) for
bug reports and feature requests.
