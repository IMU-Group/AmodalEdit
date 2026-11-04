# [AmodalEdit: Advancing Instruction-Based Image Editing via Amodal Scene Layout Perception]()

Official implementation of the following paper:

AmodalEdit: Advancing Instruction-Based Image Editing via Amodal Scene Layout Perception, ACM MM 2026

[[`Paper`](https://extionkj.github.io/AmodalEdit/static/pdfs/paper.pdf)] [[`Project`](https://extionkj.github.io/AmodalEdit/)]

![![title]](imgs/teaser.png?raw=true)

## Installation
We suggest users to use the conda for creating new python environment.
- This project uses [CLIP](https://github.com/openai/CLIP), [timm](https://github.com/huggingface/pytorch-image-models) and [Diffusers](https://github.com/huggingface/diffusers) for feature extraction and image generation.
- `pip install -r requirements.txt`

### Example conda environment setup
```bash
git clone https://github.com/IMU-Group/AmodalEdit.git
cd amodaledit
conda create -n amodaledit python=3.10
conda activate amodaledit

pip install -r requirements.txt
```

## Run Experiments

```bash
conda activate amodaledit
python train_alp.py --config configs/alp_stage1.yaml
```

Stage 2 trains the IAA on instruction-editing triplets, starting from the stage-1 checkpoint:

```bash
python train_edit.py --config configs/edit_stage2.yaml \
    --alp-checkpoint runs/alp_stage1/best.pth
```

Inference reads a cached instruction graph produced by `tools/run_igc.py`:

```bash
python infer.py --config configs/edit_stage2.yaml \
    --image demo.jpg --instruction "Move the vase to the right of the book" \
    --igc-json runs/igc/demo.json --key demo \
    --alp-checkpoint runs/alp_stage1/best.pth \
    --iaa-checkpoint runs/edit_stage2/last.pth \
    --output outputs/edited.png
```
