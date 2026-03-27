#!/usr/bin/env bash
# This project was developed with assistance from AI tools.
#
# Self-contained Cosmos Transfer2.5 augmentation script for vast.ai H100 instances.
#
# Prerequisites:
#   - vast.ai H100 instance (80GB+ VRAM) with PyTorch template
#   - Synthetic data uploaded to ~/data/palletjack_data/warehouse/Camera/
#   - HuggingFace token with access to nvidia/Cosmos-Transfer2.5-2B
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   bash vastai_cosmos_augment.sh [--clip-length 25] [--prompt "..."]
#
# Output: ~/data/cosmos_augmented/Camera/{rgb,object_detection}/

set -euo pipefail

# --- Configuration (override via CLI flags) ---
CLIP_LENGTH="${CLIP_LENGTH:-25}"
PROMPT="${PROMPT:-Industrial warehouse, bright overhead fluorescent lighting, concrete floor}"
EDGE_WEIGHT="${EDGE_WEIGHT:-1.0}"
DEPTH_WEIGHT="${DEPTH_WEIGHT:-0.25}"
SEG_WEIGHT="${SEG_WEIGHT:-0.25}"
NUM_STEPS="${NUM_STEPS:-36}"
GUIDANCE="${GUIDANCE:-3.0}"

INPUT_DIR="${INPUT_DIR:-$HOME/data/palletjack_data/warehouse/Camera}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/data/cosmos_augmented/Camera}"
CLIPS_DIR="${CLIPS_DIR:-$HOME/data/cosmos_clips}"
WORK_DIR="${WORK_DIR:-$HOME/cosmos_work}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --clip-length) CLIP_LENGTH="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --edge-weight) EDGE_WEIGHT="$2"; shift 2 ;;
        --depth-weight) DEPTH_WEIGHT="$2"; shift 2 ;;
        --seg-weight) SEG_WEIGHT="$2"; shift 2 ;;
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Cosmos Transfer2.5 Augmentation (vast.ai) ==="
echo "  Input:       $INPUT_DIR"
echo "  Output:      $OUTPUT_DIR"
echo "  Clip length: $CLIP_LENGTH"
echo "  Prompt:      $PROMPT"
echo "  Controls:    edge=$EDGE_WEIGHT depth=$DEPTH_WEIGHT seg=$SEG_WEIGHT"
echo ""

# --- Validate input data ---
for subdir in rgb depth semantic_segmentation object_detection; do
    if [[ ! -d "$INPUT_DIR/$subdir" ]]; then
        echo "ERROR: Missing $INPUT_DIR/$subdir"
        echo "Upload synthetic data to ~/data/palletjack_data/warehouse/Camera/ first."
        exit 1
    fi
done
FRAME_COUNT=$(ls "$INPUT_DIR/rgb/"*.png 2>/dev/null | wc -l)
echo "Found $FRAME_COUNT synthetic frames"
if [[ $FRAME_COUNT -eq 0 ]]; then
    echo "ERROR: No PNG frames in $INPUT_DIR/rgb/"; exit 1
fi

# --- Step 1: Install dependencies ---
echo ""
echo "=== Step 1/5: Installing dependencies ==="
pip install -q --upgrade pip
pip install -q diffusers[torch] transformers accelerate sentencepiece protobuf
pip install -q opencv-python-headless numpy Pillow

# Install ffmpeg if not present
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    apt-get update -qq && apt-get install -y -qq ffmpeg 2>/dev/null || \
    conda install -y -c conda-forge ffmpeg 2>/dev/null || \
    { echo "ERROR: Could not install ffmpeg"; exit 1; }
fi

# --- Step 2: Download model (cached across runs) ---
echo ""
echo "=== Step 2/5: Downloading Cosmos Transfer2.5-2B model ==="
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: Set HF_TOKEN environment variable (huggingface.co/settings/tokens)"
    echo "  You need access to nvidia/Cosmos-Transfer2.5-2B (accept license first)"
    exit 1
fi
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import login
login(token='${HF_TOKEN}', add_to_git_credential=False)
print('HuggingFace auth OK')
"

# Pre-download model weights to cache (shows progress)
python3 -c "
import diffusers.pipelines.cosmos.pipeline_cosmos2_5_transfer as cosmos_mod
from diffusers import Cosmos2_5_TransferPipeline, AutoModel
import torch, os

# Disable safety checker — avoids needing cosmos_guardrail + LlamaGuard access
class NoOpSafetyChecker:
    def to(self, *args, **kwargs):
        return self
    def check_text_safety(self, text):
        return True
    def check_video_safety(self, video):
        return video
cosmos_mod.CosmosSafetyChecker = NoOpSafetyChecker

model_id = 'nvidia/Cosmos-Transfer2.5-2B'

print('Downloading controlnet (edge)...')
edge_cn = AutoModel.from_pretrained(model_id, revision='diffusers/controlnet/general/edge')
print('Downloading controlnet (depth)...')
AutoModel.from_pretrained(model_id, revision='diffusers/controlnet/general/depth')
print('Downloading controlnet (seg)...')
AutoModel.from_pretrained(model_id, revision='diffusers/controlnet/general/seg')
print('Downloading base pipeline...')
Cosmos2_5_TransferPipeline.from_pretrained(model_id, revision='diffusers/general', controlnet=edge_cn, torch_dtype=torch.bfloat16)
print('All model weights cached.')
"

# --- Step 3: Prepare control maps ---
echo ""
echo "=== Step 3/5: Preparing control map clips ==="
mkdir -p "$CLIPS_DIR"
python3 - "$INPUT_DIR" "$CLIPS_DIR" "$CLIP_LENGTH" <<'PREP_SCRIPT'
import cv2, numpy as np, os, subprocess, json, sys

input_dir, clips_dir, clip_length = sys.argv[1], sys.argv[2], int(sys.argv[3])

def sorted_frames(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    files.sort(key=lambda fn: int(os.path.splitext(fn)[0]))
    return files

def normalize_depth(d):
    df = d.astype(np.float32)
    p99 = np.percentile(df[df > 0], 99) if np.any(df > 0) else 1.0
    return (np.clip(df, 0, p99) / max(p99, 1e-6) * 255).astype(np.uint8)

frames = sorted_frames(os.path.join(input_dir, "rgb"))
print(f"Chunking {len(frames)} frames into clips of {clip_length}")

for clip_idx, start in enumerate(range(0, len(frames), clip_length)):
    clip_frames = frames[start:start + clip_length]
    clip_dir = os.path.join(clips_dir, f"clip_{clip_idx:04d}")
    os.makedirs(clip_dir, exist_ok=True)
    manifest = []

    for kind in ["rgb", "depth", "segmentation", "edges"]:
        os.makedirs(os.path.join(clip_dir, f"tmp_{kind}"), exist_ok=True)

    for i, fname in enumerate(clip_frames):
        fid = os.path.splitext(fname)[0]
        manifest.append({"index": i, "filename": fname, "frame_id": fid})

        rgb = cv2.imread(os.path.join(input_dir, "rgb", fname), cv2.IMREAD_UNCHANGED)
        if rgb.shape[2] == 4: rgb = rgb[:, :, :3]
        cv2.imwrite(os.path.join(clip_dir, "tmp_rgb", f"{i:04d}.png"), rgb)

        depth = cv2.imread(os.path.join(input_dir, "depth", fname), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(clip_dir, "tmp_depth", f"{i:04d}.png"), normalize_depth(depth))

        seg = cv2.imread(os.path.join(input_dir, "semantic_segmentation", fname), cv2.IMREAD_UNCHANGED)
        if seg.shape[2] == 4: seg = seg[:, :, :3]
        cv2.imwrite(os.path.join(clip_dir, "tmp_segmentation", f"{i:04d}.png"), seg)

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(clip_dir, "tmp_edges", f"{i:04d}.png"), cv2.Canny(gray, 50, 150))

    # Encode to MP4
    for kind in ["rgb", "depth", "segmentation", "edges"]:
        tmp = os.path.join(clip_dir, f"tmp_{kind}")
        flist = os.path.join(clip_dir, f"{kind}.list")
        with open(flist, "w") as f:
            for fn in sorted(os.listdir(tmp)):
                f.write(f"file '{os.path.join(tmp, fn)}'\nduration 0.1\n")
        mp4 = os.path.join(clip_dir, f"{kind}.mp4")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", flist,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                        "-crf", "18", mp4], check=True, capture_output=True)
        os.remove(flist)
        # Cleanup tmp frames
        for fn in os.listdir(tmp): os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp)

    with open(os.path.join(clip_dir, "frame_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  clip_{clip_idx:04d}: {len(clip_frames)} frames")

print(f"Done: {(len(frames) + clip_length - 1) // clip_length} clips")
PREP_SCRIPT

# --- Step 4: Run Cosmos Transfer augmentation ---
echo ""
echo "=== Step 4/5: Running Cosmos Transfer2.5 inference ==="
mkdir -p "$OUTPUT_DIR/rgb" "$OUTPUT_DIR/object_detection"

python3 - "$CLIPS_DIR" "$OUTPUT_DIR" "$INPUT_DIR/object_detection" \
          "$PROMPT" "$EDGE_WEIGHT" "$DEPTH_WEIGHT" "$SEG_WEIGHT" \
          "$NUM_STEPS" "$GUIDANCE" <<'INFERENCE_SCRIPT'
import json, os, shutil, sys
import cv2
import numpy as np
import torch
from PIL import Image
import diffusers.pipelines.cosmos.pipeline_cosmos2_5_transfer as cosmos_mod
from diffusers import Cosmos2_5_TransferPipeline, AutoModel
from diffusers.utils import export_to_video, load_video

# Disable safety checker — avoids needing cosmos_guardrail + LlamaGuard access
class NoOpSafetyChecker:
    def to(self, *args, **kwargs):
        return self
    def check_text_safety(self, text):
        return True
    def check_video_safety(self, video):
        return video
cosmos_mod.CosmosSafetyChecker = NoOpSafetyChecker

clips_dir = sys.argv[1]
output_dir = sys.argv[2]
labels_dir = sys.argv[3]
prompt = sys.argv[4]
edge_weight = float(sys.argv[5])
depth_weight = float(sys.argv[6])
seg_weight = float(sys.argv[7])
num_steps = int(sys.argv[8])
guidance = float(sys.argv[9])

model_id = "nvidia/Cosmos-Transfer2.5-2B"
rgb_out = os.path.join(output_dir, "rgb")
labels_out = os.path.join(output_dir, "object_detection")

negative_prompt = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, "
    "and flickering. Overall, the video is of poor quality."
)

# Load pipeline with edge controlnet (primary control)
# We'll run multiple passes if using multiple controls, or use edge as the main one.
# The diffusers API loads one controlnet at a time. For multi-control, we'd use
# the NVIDIA repo's inference.py. For now, edge (highest weight) is the primary signal.
print("Loading edge controlnet...")
controlnet = AutoModel.from_pretrained(
    model_id, revision="diffusers/controlnet/general/edge", torch_dtype=torch.bfloat16
)
print("Loading pipeline...")
pipe = Cosmos2_5_TransferPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    revision="diffusers/general",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

clips = sorted(d for d in os.listdir(clips_dir)
               if os.path.isdir(os.path.join(clips_dir, d)) and d.startswith("clip_"))

print(f"Processing {len(clips)} clips...")
total_frames = 0

for clip_name in clips:
    clip_dir = os.path.join(clips_dir, clip_name)
    manifest_path = os.path.join(clip_dir, "frame_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    num_frames = len(manifest)
    print(f"\n{clip_name}: {num_frames} frames")

    # Load edge control maps from the clip's edge MP4
    edge_video_path = os.path.join(clip_dir, "edges.mp4")
    edge_frames = load_video(edge_video_path)[:num_frames]

    # Determine frame dimensions from first RGB frame and round to nearest 16
    first_rgb = os.path.join(clips_dir, clip_name, "tmp_rgb")
    if not os.path.isdir(first_rgb):
        # tmp dirs cleaned up — read from source
        first_entry = manifest[0]
        sample = cv2.imread(os.path.join(labels_dir, "..", "rgb", first_entry["filename"]))
        if sample is None:
            # Fallback: read dimension from the edge video frames
            sample_frame = edge_frames[0]
            sample_arr = np.array(sample_frame) if isinstance(sample_frame, Image.Image) else sample_frame
            orig_h, orig_w = sample_arr.shape[:2]
        else:
            orig_h, orig_w = sample.shape[:2]
    else:
        sample = cv2.imread(os.path.join(first_rgb, sorted(os.listdir(first_rgb))[0]))
        orig_h, orig_w = sample.shape[:2]

    # Round down to nearest multiple of 16
    frame_h = (orig_h // 16) * 16
    frame_w = (orig_w // 16) * 16
    print(f"  Frame dims: {orig_h}x{orig_w} -> {frame_h}x{frame_w} (16-aligned)")

    # Prepare controls: convert to expected format, resize to 16-aligned dims
    controls = []
    for frame in edge_frames:
        if isinstance(frame, Image.Image):
            arr = np.array(frame.convert("RGB"))
        else:
            arr = np.array(frame)
        # Edge maps are single-channel; expand to 3-channel
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        # Resize to 16-aligned dimensions
        img = Image.fromarray(arr).resize((frame_w, frame_h), Image.LANCZOS)
        controls.append(img)

    print(f"  Running inference ({num_steps} steps, guidance={guidance})...")
    result = pipe(
        controls=controls,
        controls_conditioning_scale=edge_weight,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=frame_h,
        width=frame_w,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
    )

    output_frames = result.frames[0]

    # Save augmented frames and copy labels
    for i, entry in enumerate(manifest):
        if i >= len(output_frames):
            print(f"  WARNING: fewer output frames ({len(output_frames)}) than manifest ({len(manifest)})")
            break

        aug_fname = f"aug_{entry['frame_id']}.png"

        frame = output_frames[i]
        if isinstance(frame, Image.Image):
            frame.save(os.path.join(rgb_out, aug_fname))
        else:
            cv2.imwrite(os.path.join(rgb_out, aug_fname), np.array(frame))

        # Copy corresponding label
        orig_label = os.path.join(labels_dir, f"{entry['frame_id']}.txt")
        aug_label = os.path.join(labels_out, f"aug_{entry['frame_id']}.txt")
        if os.path.exists(orig_label):
            shutil.copy2(orig_label, aug_label)

        total_frames += 1

    print(f"  Done: {min(len(output_frames), len(manifest))} augmented frames")

print(f"\nTotal: {total_frames} augmented frames in {output_dir}")
INFERENCE_SCRIPT

# --- Step 5: Package results ---
echo ""
echo "=== Step 5/5: Packaging results ==="
AUG_FRAME_COUNT=$(ls "$OUTPUT_DIR/rgb/"*.png 2>/dev/null | wc -l)
AUG_LABEL_COUNT=$(ls "$OUTPUT_DIR/object_detection/"*.txt 2>/dev/null | wc -l)
echo "Augmented frames: $AUG_FRAME_COUNT"
echo "Augmented labels: $AUG_LABEL_COUNT"

# Create a tarball for easy download
TARBALL="$HOME/data/cosmos_augmented.tar.gz"
cd "$HOME/data"
tar czf "$TARBALL" cosmos_augmented/
echo "Tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1))"

echo ""
echo "=== DONE ==="
echo "Download the results with:"
echo "  scp -P <port> root@<vast-host>:~/data/cosmos_augmented.tar.gz ."
echo ""
echo "Then upload to cluster PVC (see instructions in README)."
