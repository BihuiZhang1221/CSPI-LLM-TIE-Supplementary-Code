# Physics-Guided Semantic Gating With Embedded Language Models for Real-Time AUV Control in Subsea Infrastructure Inspection

Supplementary code for the manuscript "Physics-Guided Semantic Control of AUVs
Using Embedded Language Models for Real-Time Subsea Inspection" (submitted to
IEEE Transactions on Industrial Electronics). The proposed framework is
referred to as CSPI-LLM.

## Contents

- `src/teacher/teacher.py` - teacher prompts, physics constraints, student interface
- `src/data/dataset.py` - teacher labeling stages and compact-JSON SFT build
- `src/sac/semantic_sac.py` - semantic-gated SAC policy and replay buffer
- `src/hardware/hardware.py` - thruster allocation, PWM calibration, state estimator
- `src/isaacsim/` - Isaac Lab environment code (reference; requires Isaac Sim)
- `src/oceansim/` - OceanSim sensor utilities (BSD-3-Clause; see `src/oceansim/LICENSE`)
- `supplementary/` - supplementary material PDF

## Quick start

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...
python -m src.data.dataset --help
python -m src.data.dataset build --output-dir datasets/teacher_labels
python -m src.data.dataset prompts --output-dir datasets/teacher_labels
python -m src.data.dataset run --output-dir datasets/teacher_labels
python -m src.data.dataset parse --output-dir datasets/teacher_labels
python -m src.data.dataset audit --output-dir datasets/teacher_labels
python -m src.data.dataset sft
```
