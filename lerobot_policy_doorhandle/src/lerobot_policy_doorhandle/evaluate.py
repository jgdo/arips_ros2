"""Offline evaluation of a doorhandle checkpoint on a LeRobot dataset.

usage:
  python -m lerobot_policy_doorhandle.evaluate --policy CHECKPOINT/pretrained_model \
      --dataset-root ROOT [--episodes 0 1 2] [--batch-size 32] [--json out.json]

Prints per-joint mean absolute error (slider in mm, rotary joints in degrees), the door gripper
position error from forward kinematics (mm), the mean predicted σ, and the error at the first and
last frame of every episode (the first frame is the hardest: the arm is still far from the target).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION

from .kinematics import gripper_position
from .modeling_doorhandle import FK_JOINTS, DoorhandlePolicy


def _unit(name: str, config) -> tuple[float, str]:
    """Display factor / unit: prismatic joints (with a norm range) in mm, the rest in degrees."""
    return (1000.0, "mm") if name in config.joint_norm_ranges else (180.0 / math.pi, "deg")


def evaluate(policy_path: str, dataset_root: str, episodes: list[int] | None = None, batch_size: int = 32,
             device: str | None = None) -> dict:
    policy = DoorhandlePolicy.from_pretrained(policy_path)
    if device:
        policy.config.device = device
    policy.to(policy.config.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=policy_path)

    root = Path(dataset_root)
    dataset = LeRobotDataset(f"local/{root.name}", root=root, episodes=episodes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    preds, sigmas, targets, ep_idx, frame_idx = [], [], [], [], []
    for batch in loader:
        targets.append(batch[ACTION].float())
        ep_idx.append(batch["episode_index"])
        frame_idx.append(batch["frame_index"])
        obs = preprocessor({k: v for k, v in batch.items() if k != ACTION})
        mean, sigma = policy.predict_with_uncertainty(obs)
        preds.append(postprocessor(mean).cpu().float())
        sigmas.append(sigma.cpu().float())
    pred, sigma, target = torch.cat(preds), torch.cat(sigmas), torch.cat(targets)
    ep_idx, frame_idx = torch.cat(ep_idx), torch.cat(frame_idx)

    cfg = policy.config
    names = list(cfg.action_joint_names)
    err = (pred - target).abs()
    result = {"frames": len(pred), "episodes": sorted(set(ep_idx.tolist())), "joints": {}}
    for j, name in enumerate(names):
        f, unit = _unit(name, cfg)
        result["joints"][name] = {"unit": unit, "mae": err[:, j].mean().item() * f,
                                  "sigma": sigma[:, j].mean().item() * f}
    fk = None
    if all(j in names for j in FK_JOINTS):
        idx = [names.index(j) for j in FK_JOINTS]
        fk = (gripper_position(*pred[:, idx].unbind(-1)) - gripper_position(*target[:, idx].unbind(-1)))
        fk = fk.norm(dim=-1) * 1000.0
        result["fk_mm"] = {"mean": fk.mean().item(), "median": fk.median().item(), "max": fk.max().item()}

    per_episode = []
    for ep in result["episodes"]:
        mask = ep_idx == ep
        first = mask.nonzero()[frame_idx[mask].argmin()].item()
        last = mask.nonzero()[frame_idx[mask].argmax()].item()
        row = {"episode": ep, "frames": int(mask.sum())}
        if fk is not None:
            row.update(fk_first_mm=fk[first].item(), fk_last_mm=fk[last].item(), fk_mean_mm=fk[mask].mean().item())
        per_episode.append(row)
    result["per_episode"] = per_episode
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="checkpoint folder with config.json + model.safetensors")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episodes", type=int, nargs="*", help="episode indices (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="override the device stored in the checkpoint config")
    parser.add_argument("--json", type=Path, help="also write the result to this file")
    args = parser.parse_args(argv)

    r = evaluate(args.policy, args.dataset_root, args.episodes, args.batch_size, args.device)
    print(f"{r['frames']} frames, {len(r['episodes'])} episodes")
    for name, j in r["joints"].items():
        print(f"  {name:20s} MAE {j['mae']:8.2f} {j['unit']:3s}  mean sigma {j['sigma']:8.2f} {j['unit']}")
    if "fk_mm" in r:
        f = r["fk_mm"]
        print(f"  gripper position error: mean {f['mean']:.1f} mm, median {f['median']:.1f} mm, max {f['max']:.1f} mm")
        print("  episode  frames  first[mm]  last[mm]  mean[mm]")
        for e in r["per_episode"]:
            print(f"  {e['episode']:7d}  {e['frames']:6d}  {e['fk_first_mm']:9.1f}  {e['fk_last_mm']:8.1f}  "
                  f"{e['fk_mean_mm']:8.1f}")
    if args.json:
        args.json.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
