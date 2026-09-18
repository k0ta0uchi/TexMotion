"""Native TexMotion port of WHAM's offline temporal inference stages.

The module ports the checkpoint-compatible ``MotionEncoder``,
``NeuralInitialization``, ``Regressor``, image ``Integrator``,
``TrajectoryDecoder``, ``MotionDecoder``, and ``TrajectoryRefiner`` layers.
It uses the same layer names, dimensions, and recurrent rollout as the
upstream model, so ``wham_vit_w_3dpw.pth.tar`` can run without importing the
research repository.  Neutral SMPL buffers embedded in the checkpoint are
also used for first-frame initialization and LBS-based SMPL decoding.

ViTPose/DPVO preprocessing and image-feature extraction remain explicit local
asset boundaries.  TexMotion supplies a local RTMPose 2D detector when its
ONNX weights are present, with MediaPipe as an offline fallback, and records
which optional stages were unavailable.  WHAM's released camera-space output
(Y-down/Z-away) is canonicalized to SMPL-X (Y-up/Z-forward) once at the
adapter boundary.  Projects with the full research assets can still select a
custom implementation through ``TEXMOTION_WHAM_ADAPTER_IMPL``.

Source: https://github.com/yohanshin/WHAM (MIT License, Soyong Shin, 2023).
The layer definitions below preserve the upstream architecture and parameter
names; the surrounding adapter code is TexMotion integration code.

Upstream MIT notice: Copyright (c) 2023 Soyong Shin. Permission is hereby
granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: the above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Keep the default MediaPipe runtime torch-free at import time.
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised by the lightweight profile
    torch = None
    nn = None


class _UnverifiedFeatureSourceError(ValueError):
    """A raw model checkpoint was supplied where frame features were required."""


# MediaPipe's canonical topology to the 17 joints consumed by WHAM.  The
# input detector and native motion encoder are both COCO order (nose, eyes,
# ears, shoulders, elbows, wrists, hips, knees, ankles).  External quality
# adapters may advertise the separate common/H36M ``wham_j17`` ordering; the
# generic backend keeps that mapping for compatibility.
COCO17_TO_MEDIAPIPE = {
    0: 0,
    1: 2,
    2: 5,
    3: 7,
    4: 8,
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
}


if nn is not None:

    class Regressor(nn.Module):
        """WHAM's recurrent decoder, with the original parameter names."""

        def __init__(
            self,
            in_dim: int,
            hid_dim: int,
            out_dims: Sequence[int],
            init_dim: int,
            layer: str = "LSTM",
            n_layers: int = 2,
            n_iters: int = 1,
        ) -> None:
            super().__init__()
            del n_iters  # retained for checkpoint/API compatibility
            self.n_outs = len(out_dims)
            rnn_cls = getattr(nn, layer.upper())
            self.rnn = rnn_cls(
                in_dim + init_dim,
                hid_dim,
                n_layers,
                bidirectional=False,
                batch_first=True,
                dropout=0.3,
            )
            for i, out_dim in enumerate(out_dims):
                layer_module = nn.Linear(hid_dim, out_dim)
                # This is the initialization used by upstream WHAM.  Loading
                # a checkpoint replaces these values, while preserving the
                # same state-dict shape when a small fixture is used.
                nn.init.xavier_uniform_(layer_module.weight, gain=0.01)
                setattr(self, f"declayer{i}", layer_module)

        def forward(self, x: "torch.Tensor", inits: Sequence["torch.Tensor"], h0: Any):
            xc = torch.cat([x, *inits], dim=-1)
            xc, h0 = self.rnn(xc, h0)
            preds = [getattr(self, f"declayer{j}")(xc) for j in range(self.n_outs)]
            return preds, xc, h0


    class NeuralInitialization(nn.Module):
        """Maps the first-frame initialization to recurrent hidden state."""

        def __init__(self, in_dim: int, hid_dim: int, layer: str, n_layers: int) -> None:
            super().__init__()
            self.n_layers = n_layers
            self.num_inits = int(layer.upper() == "LSTM") + 1
            out_dim = hid_dim * self.num_inits * n_layers
            self.linear1 = nn.Linear(in_dim, hid_dim)
            self.linear2 = nn.Linear(hid_dim, hid_dim * n_layers)
            self.linear3 = nn.Linear(hid_dim * n_layers, out_dim)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()

        def forward(self, x: "torch.Tensor"):
            b = x.shape[0]
            out = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(x)))))
            out = out.view(b, self.num_inits, self.n_layers, -1)
            out = out.permute(1, 2, 0, 3).contiguous()
            if self.num_inits == 2:
                return tuple(out)
            return out[0]


    class MotionEncoder(nn.Module):
        """Port of ``lib.models.layers.modules.MotionEncoder`` from WHAM."""

        def __init__(
            self,
            in_dim: int,
            d_embed: int,
            pose_dr: float,
            rnn_type: str,
            n_layers: int,
            n_joints: int,
        ) -> None:
            super().__init__()
            self.n_joints = n_joints
            self.embed_layer = nn.Linear(in_dim, d_embed)
            self.pos_drop = nn.Dropout(pose_dr)
            self.neural_init = NeuralInitialization(
                n_joints * 3 + in_dim, d_embed, rnn_type, n_layers
            )
            self.regressor = Regressor(
                d_embed,
                d_embed,
                [n_joints * 3],
                n_joints * 3,
                rnn_type,
                n_layers,
            )

        def forward(self, x: "torch.Tensor", init: "torch.Tensor"):
            b, f = x.shape[:2]
            x = self.embed_layer(x.reshape(b, f, -1))
            x = self.pos_drop(x)
            h0 = self.neural_init(init)
            pred_list = [init[..., : self.n_joints * 3]]
            motion_context_list = []

            for i in range(f):
                (pred_kp3d,), motion_context, h0 = self.regressor(
                    x[:, [i]], pred_list[-1:], h0
                )
                motion_context_list.append(motion_context)
                pred_list.append(pred_kp3d)

            pred_kp3d = torch.cat(pred_list[1:], dim=1).view(b, f, -1, 3)
            motion_context = torch.cat(motion_context_list, dim=1)
            motion_context = torch.cat(
                (motion_context, pred_kp3d.reshape(b, f, -1)), dim=-1
            )
            return pred_kp3d, motion_context


    class NativeWhamNetwork(nn.Module):
        """Checkpoint-compatible wrapper for WHAM's motion encoder."""

        def __init__(
            self,
            n_joints: int = 17,
            in_dim: int = 37,
            d_embed: int = 512,
            pose_dr: float = 0.15,
            rnn_type: str = "LSTM",
            n_layers: int = 3,
            d_feat: int = 1024,
            enable_decoders: bool = False,
        ) -> None:
            super().__init__()
            self.n_joints = n_joints
            self.in_dim = in_dim
            self.d_embed = d_embed
            self.rnn_type = rnn_type
            self.n_layers = n_layers
            self.d_feat = d_feat
            # Upstream Network owns this parameter.  It is initialized to zero
            # in the official model and therefore requires no extra behavior
            # for TexMotion's binary visibility mask.
            self.mask_embedding = nn.Parameter(torch.zeros(1, 1, n_joints, 2))
            self.motion_encoder = MotionEncoder(
                in_dim=in_dim,
                d_embed=d_embed,
                pose_dr=pose_dr,
                rnn_type=rnn_type,
                n_layers=n_layers,
                n_joints=n_joints,
            )
            if enable_decoders:
                self._attach_official_decoders(d_feat)

        def _attach_official_decoders(self, d_feat: int) -> None:
            """Attach the remaining official WHAM stages lazily.

            The compact motion-only path remains import-compatible with old
            fixtures.  Looking up these classes at instance construction time
            lets the torch-free import path keep working while still exposing
            the exact official state-dict names when a full checkpoint is
            loaded.
            """
            d_context = self.d_embed + self.n_joints * 3
            self.trajectory_decoder = TrajectoryDecoder(
                d_embed=d_context,
                rnn_type=self.rnn_type,
                n_layers=self.n_layers,
            )
            self.integrator = Integrator(
                in_channel=d_feat + d_context,
                out_channel=d_context,
            )
            self.motion_decoder = MotionDecoder(
                d_embed=d_context,
                rnn_type=self.rnn_type,
                n_layers=self.n_layers,
            )
            self.trajectory_refiner = TrajectoryRefiner(
                d_embed=d_context,
                d_hidden=self.d_embed,
                rnn_type=self.rnn_type,
                n_layers=2,
            )

        def preprocess(self, x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            """Apply WHAM's masked-keypoint preprocessing exactly."""
            b, f = x.shape[:2]
            mask = mask.to(dtype=x.dtype)
            mask_embedding = mask.unsqueeze(-1) * self.mask_embedding
            flat_mask = mask.unsqueeze(-1).repeat(1, 1, 1, 2).reshape(b, f, -1)
            flat_mask = torch.cat((flat_mask, torch.zeros_like(flat_mask[..., :3])), dim=-1)
            flat_embedding = mask_embedding.reshape(b, f, -1)
            flat_embedding = torch.cat(
                (flat_embedding, torch.zeros_like(flat_embedding[..., :3])), dim=-1
            )
            # The official code masks in-place.  Clone in the caller so a
            # caller can safely reuse the normalized 2D sequence.
            x = x.clone()
            x[flat_mask.bool()] = 0.0
            return x + flat_embedding

else:  # pragma: no cover - import-only fallback when torch is absent
    Regressor = NeuralInitialization = MotionEncoder = NativeWhamNetwork = None


if nn is not None:

    class Integrator(nn.Module):
        """WHAM's optional image-feature integrator.

        This is a literal port of ``lib.models.layers.modules.Integrator``.
        In particular, an all-zero feature row is treated as missing and does
        not receive the residual connection.  Keeping that detail matters for
        callers that provide sparse/offline feature archives.
        """

        def __init__(self, in_channel: int, out_channel: int, hid_channel: int = 1024) -> None:
            super().__init__()
            self.layer1 = nn.Linear(in_channel, hid_channel)
            self.relu1 = nn.ReLU()
            self.dr1 = nn.Dropout(0.1)
            self.layer2 = nn.Linear(hid_channel, hid_channel)
            self.relu2 = nn.ReLU()
            self.dr2 = nn.Dropout(0.1)
            self.layer3 = nn.Linear(hid_channel, out_channel)

        def forward(self, x: "torch.Tensor", feat: "torch.Tensor") -> "torch.Tensor":
            residual = x
            mask = (feat != 0).all(dim=-1).all(dim=-1)
            out = torch.cat((x, feat), dim=-1)
            out = self.dr1(self.relu1(self.layer1(out)))
            out = self.dr2(self.relu2(self.layer2(out)))
            out = self.layer3(out)
            out[mask] = out[mask] + residual[mask]
            return out


    class TrajectoryDecoder(nn.Module):
        """Checkpoint-compatible port of WHAM's global trajectory decoder."""

        def __init__(self, d_embed: int, rnn_type: str, n_layers: int) -> None:
            super().__init__()
            self.regressor = Regressor(
                d_embed,
                d_embed,
                [3, 6],
                12,
                rnn_type,
                n_layers,
            )

        def forward(
            self,
            x: "torch.Tensor",
            root: "torch.Tensor",
            cam_a: "torch.Tensor",
            h0: Any = None,
        ):
            b, f = x.shape[:2]
            pred_root_list, pred_vel_list = [root[:, :1]], []
            for i in range(f):
                (pred_rootv, pred_rootr), _, h0 = self.regressor(
                    x[:, [i]], [pred_root_list[-1], cam_a[:, [i]]], h0
                )
                pred_root_list.append(pred_rootr)
                pred_vel_list.append(pred_rootv)
            pred_root = torch.cat(pred_root_list, dim=1).view(b, f + 1, -1)
            pred_vel = torch.cat(pred_vel_list, dim=1).view(b, f, -1)
            return pred_root, pred_vel


    class MotionDecoder(nn.Module):
        """Checkpoint-compatible port of WHAM's SMPL motion decoder."""

        # This list is configs.constants.BMODEL.MAIN_JOINTS in the official
        # repository.  It is part of the decoder's parameter shape contract.
        MAIN_JOINTS = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
        )

        def __init__(self, d_embed: int, rnn_type: str, n_layers: int) -> None:
            super().__init__()
            self.n_pose = 24
            self.neural_init = NeuralInitialization(
                len(self.MAIN_JOINTS) * 6,
                d_embed,
                rnn_type,
                n_layers,
            )
            self.regressor = Regressor(
                d_embed,
                d_embed,
                [self.n_pose * 6, 10, 3, 4],
                self.n_pose * 6,
                rnn_type,
                n_layers,
            )

        def forward(self, x: "torch.Tensor", init: "torch.Tensor"):
            b, f = x.shape[:2]
            h0 = self.neural_init(
                init[:, :, list(self.MAIN_JOINTS)].reshape(b, 1, -1)
            )
            pred_pose_list = [init.reshape(b, 1, -1)]
            pred_shape_list, pred_cam_list, pred_contact_list = [], [], []
            for i in range(f):
                (pred_pose, pred_shape, pred_cam, pred_contact), _, h0 = self.regressor(
                    x[:, [i]], pred_pose_list[-1:], h0
                )
                pred_pose_list.append(pred_pose)
                pred_shape_list.append(pred_shape)
                pred_cam_list.append(pred_cam)
                pred_contact_list.append(pred_contact)
            pred_pose = torch.cat(pred_pose_list[1:], dim=1).view(b, f, -1)
            pred_shape = torch.cat(pred_shape_list, dim=1).view(b, f, -1)
            pred_cam = torch.cat(pred_cam_list, dim=1).view(b, f, -1)
            pred_contact = torch.cat(pred_contact_list, dim=1).view(b, f, -1)
            return pred_pose, pred_shape, pred_cam, pred_contact


    class TrajectoryRefiner(nn.Module):
        """Checkpoint-compatible port of WHAM's foot-contact refiner."""

        def __init__(
            self,
            d_embed: int,
            d_hidden: int,
            rnn_type: str,
            n_layers: int,
        ) -> None:
            super().__init__()
            self.refiner = Regressor(
                d_embed + 12,
                d_hidden,
                [6, 3],
                9,
                rnn_type,
                n_layers,
            )

        def forward(
            self,
            context: "torch.Tensor",
            pred_vel: "torch.Tensor",
            output: Mapping[str, "torch.Tensor"],
            cam_angvel: "torch.Tensor",
            return_y_up: bool = True,
        ) -> Dict[str, "torch.Tensor"]:
            # The upstream implementation computes a contact-weighted foot
            # velocity and receives the already-decoded SMPL feet in output.
            # Keep this method self-contained; the adapter only calls it when
            # an embedded/external SMPL decoder supplied those fields.
            del cam_angvel, return_y_up
            b, f = context.shape[:2]
            pred_root = output["poses_root_r6d"].clone().detach()
            feet = output["feet"].clone().detach()
            contact = output["contact"].clone().detach()
            feet_vel = torch.cat(
                (torch.zeros_like(feet[:, :1]), feet[:, 1:] - feet[:, :-1]),
                dim=1,
            ) * 30.0
            foot_input = (feet_vel * contact.unsqueeze(-1)).reshape(b, f, -1)
            inpt_feat = torch.cat([context, foot_input], dim=-1)
            (delta_root, delta_vel), _, _ = self.refiner(
                inpt_feat,
                [pred_root[:, 1:], pred_vel],
                h0=None,
            )
            pred_root[:, 1:] = pred_root[:, 1:] + delta_root
            output = dict(output)
            output.update({
                "poses_root_r6d_refined": pred_root,
                "vel_root_refined": pred_vel + delta_vel,
            })
            return output


    def _rotation_6d_to_matrix_torch(d6: "torch.Tensor") -> "torch.Tensor":
        """Small local copy of WHAM's Zhou et al. conversion."""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)


    def _matrix_to_rotation_6d_torch(matrix: "torch.Tensor") -> "torch.Tensor":
        return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


    def _rollout_global_motion_torch(
        root_r: "torch.Tensor",
        root_v: "torch.Tensor",
        init_trans: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Port of ``rollout_global_motion`` from WHAM's layer utils."""
        root = _rotation_6d_to_matrix_torch(root_r)
        vel_world = (root[:, :-1] @ root_v.unsqueeze(-1)).squeeze(-1)
        trans = torch.cumsum(vel_world, dim=1)
        if init_trans is not None:
            trans = trans + init_trans
        return root[:, 1:], trans


    class EmbeddedSMPL(nn.Module):
        """Minimal SMPL forward pass backed by tensors in WHAM checkpoints.

        The official checkpoint stores the neutral SMPL buffers under
        ``smpl.*``.  Using those buffers avoids importing ``smplx`` or asking
        the offline TexMotion installation to download a separately licensed
        body-model asset.  It intentionally implements only inference fields
        needed by WHAM (joints, feet, vertices and full pose).
        """

        def __init__(self, state: Mapping[str, "torch.Tensor"]) -> None:
            super().__init__()
            required = (
                "shapedirs",
                "v_template",
                "J_regressor",
                "posedirs",
                "lbs_weights",
                "parents",
                "J_regressor_wham",
            )
            missing = [name for name in required if name not in state]
            if missing:
                raise ValueError("embedded WHAM SMPL state is missing: " + ", ".join(missing))
            for name in required:
                self.register_buffer(name, state[name].detach().float())
            if "faces_tensor" in state:
                self.register_buffer("faces_tensor", state["faces_tensor"].detach().long())
            if "J_regressor_feet" in state:
                self.register_buffer(
                    "J_regressor_feet", state["J_regressor_feet"].detach().float()
                )
            else:
                self.J_regressor_feet = None
            self.register_buffer(
                "identity",
                torch.eye(3, dtype=torch.float32),
                persistent=False,
            )

        @property
        def n_vertices(self) -> int:
            return int(self.v_template.shape[0])

        def _joint_regression(self, regressor: "torch.Tensor", vertices: "torch.Tensor"):
            return torch.einsum("jv,bvc->bjc", regressor, vertices)

        def forward(
            self,
            pose6d: "torch.Tensor",
            betas: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            b, f = pose6d.shape[:2]
            pose6d = pose6d.reshape(b * f, 24, 6)
            rot = _rotation_6d_to_matrix_torch(pose6d)
            betas = betas.reshape(b * f, -1)
            shapedirs = self.shapedirs
            if shapedirs.ndim == 2:
                shapedirs = shapedirs.reshape(self.n_vertices, 3, -1)
            v_shaped = self.v_template.unsqueeze(0) + torch.einsum(
                "vci,bi->bvc", shapedirs, betas
            )
            J = self._joint_regression(self.J_regressor, v_shaped)
            posedirs = self.posedirs
            pose_feature = (rot[:, 1:] - self.identity).reshape(b * f, -1)
            if posedirs.ndim == 2:
                pose_offsets = pose_feature @ (
                    posedirs if posedirs.shape[0] == pose_feature.shape[1] else posedirs.t()
                )
                pose_offsets = pose_offsets.reshape(b * f, self.n_vertices, 3)
            else:
                pose_offsets = torch.einsum("vci,bi->bvc", posedirs, pose_feature)
            v_posed = v_shaped + pose_offsets

            transforms: List["torch.Tensor"] = []
            parents = [int(value) for value in self.parents.detach().cpu().reshape(-1).tolist()]
            for joint_index, parent in enumerate(parents):
                relative = torch.zeros(
                    (b * f, 4, 4), dtype=rot.dtype, device=rot.device
                )
                relative[:, :3, :3] = rot[:, joint_index]
                relative[:, :3, 3] = J[:, joint_index]
                if parent >= 0:
                    relative[:, :3, 3] -= J[:, parent]
                    current = transforms[parent] @ relative
                else:
                    current = relative
                current[:, 3, 3] = 1.0
                transforms.append(current)
            transform = torch.stack(transforms, dim=1)
            transform_offset = transform.clone()
            transform_offset[:, :, :3, 3] -= J
            homogeneous = torch.cat(
                (v_posed, torch.ones((b * f, self.n_vertices, 1), device=rot.device)),
                dim=-1,
            )
            blended = torch.einsum(
                "vj,bjkl->bvkl", self.lbs_weights, transform_offset
            )
            vertices = torch.einsum("bvkl,bvl->bvk", blended, homogeneous)[..., :3]
            joints = self._joint_regression(self.J_regressor_wham, vertices)
            feet = (
                self._joint_regression(self.J_regressor_feet, vertices)
                if self.J_regressor_feet is not None
                else torch.zeros((b * f, 4, 3), device=vertices.device, dtype=vertices.dtype)
            )
            offset = joints[:, [11, 12]].mean(dim=1)
            vertices = vertices - offset[:, None]
            joints = joints - offset[:, None]
            feet = feet - offset[:, None]
            return {
                "vertices": vertices.reshape(b, f, self.n_vertices, 3),
                "joints": joints.reshape(b, f, -1, 3),
                "feet": feet.reshape(b, f, -1, 3),
                "global_orient": rot[:, :1].reshape(b, f, 1, 3, 3),
                "body_pose": rot[:, 1:].reshape(b, f, 23, 3, 3),
                "full_pose": rot.reshape(b, f, 24, 3, 3),
                "offset": offset.reshape(b, f, 3),
            }

        def neutral_initialization(self, device: "torch.device") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Return SMPL-derived neutral J17 and full-pose 6D seeds."""
            pose6d = torch.zeros((1, 1, 24, 6), device=device, dtype=torch.float32)
            pose6d[..., 0] = 1.0
            pose6d[..., 4] = 1.0
            betas = torch.zeros((1, 1, 10), device=device, dtype=torch.float32)
            output = self(pose6d, betas)
            return output["joints"][0, 0], pose6d[0, 0]

else:  # pragma: no cover - import-only fallback when torch is absent
    Integrator = TrajectoryDecoder = MotionDecoder = TrajectoryRefiner = None
    EmbeddedSMPL = None
    _rollout_global_motion_torch = None


def _safe_torch_load(torch_module: Any, path: str, device: str) -> Any:
    """Load old WHAM archives across PyTorch versions."""
    try:
        return torch_module.load(path, map_location=device, weights_only=False)
    except TypeError:  # ``weights_only`` was added after the WHAM release.
        return torch_module.load(path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    """Find a model state dictionary in common WHAM/checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        raise RuntimeError("WHAM checkpoint is not a dictionary/state-dict archive")
    candidates = ("model", "state_dict", "model_state_dict", "network", "generator")
    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            checkpoint = value
            break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise RuntimeError("WHAM checkpoint contains no tensor state dictionary")
    state: Dict[str, Any] = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            continue
        # DataParallel and common wrapper prefixes do not belong to the
        # checkpoint-compatible NativeWhamNetwork module.
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "network.", "model."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        state[normalized] = value
    if not state:
        raise RuntimeError("WHAM checkpoint contains no tensor parameters")
    return state


def _device_name(config: Dict[str, Any]) -> str:
    requested = str(config.get("device", "cpu") or "cpu").strip().lower()
    if requested in ("", "auto"):
        requested = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        return "cpu"
    return requested


def _as_keypoints2d(detector_observation: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Read canonical MediaPipe observations into ``(17,2)`` + confidence."""
    points = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros((17,), dtype=np.float32)
    if detector_observation is None:
        return points, scores
    source = getattr(detector_observation, "keypoints_2d", None)
    if not source:
        return points, scores
    # ``COCO17_TO_MEDIAPIPE`` maps the WHAM/COCO output slot to the source
    # MediaPipe-33 slot.  Keep the returned array in COCO order; indexing it by
    # the MediaPipe target slot would both swap joints and overflow this
    # 17-element array for hips/knees/ankles (23..28).
    for coco_index, mediapipe_index in COCO17_TO_MEDIAPIPE.items():
        if mediapipe_index >= len(source):
            continue
        kp = source[mediapipe_index]
        try:
            x = float(getattr(kp, "x", 0.0))
            y = float(getattr(kp, "y", 0.0))
            score = float(getattr(kp, "score", getattr(kp, "visibility", 0.0)))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        points[coco_index] = np.clip([x, y], 0.0, 1.0)
        scores[coco_index] = np.clip(score, 0.0, 1.0)
    return points, scores


def _person_bboxes_from_keypoints(
    points2d: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    frames: Sequence[np.ndarray],
) -> List[Optional[Dict[str, Any]]]:
    """Build WHAM/HMR2 center-scale person crops from detector observations.

    The released WHAM preprocessor derives a square crop from tracked 2D
    keypoints and calls HMR2 with ``encode=True``.  Keep that conversion at the
    native adapter boundary so a feature runner receives the same source-frame
    identity as the temporal core and never has to guess image dimensions.
    """

    result: List[Optional[Dict[str, Any]]] = []
    for points, point_scores, frame in zip(points2d, scores, frames):
        if frame is None or np.asarray(frame).ndim < 2:
            result.append(None)
            continue
        height, width = np.asarray(frame).shape[:2]
        values = np.asarray(points, dtype=np.float32)
        confidences = np.asarray(point_scores, dtype=np.float32).reshape(-1)
        if values.ndim != 2 or values.shape[1] < 2:
            result.append(None)
            continue
        limit = min(values.shape[0], confidences.shape[0])
        visible = (
            np.isfinite(values[:limit, :2]).all(axis=1)
            & np.isfinite(confidences[:limit])
            & (confidences[:limit] >= 0.3)
        )
        if int(np.count_nonzero(visible)) < 2:
            result.append(None)
            continue
        pixels = values[:limit, :2][visible] * np.asarray(
            [float(width), float(height)], dtype=np.float32
        )
        min_xy = pixels.min(axis=0)
        max_xy = pixels.max(axis=0)
        extent = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]))
        side = max(2.0, extent * 1.2)
        result.append({
            "cx": float((min_xy[0] + max_xy[0]) * 0.5),
            "cy": float((min_xy[1] + max_xy[1]) * 0.5),
            # WHAM/HMR2's crop helper defines side as 200 * scale.
            "scale": float(side / 200.0),
            "format": "cxcys",
            "normalized": False,
        })
    return result


def _normalize_frame(
    points_normalized: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate WHAM's deterministic bbox normalizer without SciPy.

    WHAM maps pixels into a 224x224 crop and appends three image-location
    values.  The affine form below is algebraically equivalent to that crop
    normalization and deliberately avoids random augmentation used during
    official training.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    pixels = points_normalized.astype(np.float32, copy=False) * np.array(
        [float(width), float(height)], dtype=np.float32
    )
    visible = np.isfinite(pixels).all(axis=1) & (scores >= 0.3)
    if int(visible.sum()) >= 2:
        visible_pixels = pixels[visible]
        min_xy = visible_pixels.min(axis=0)
        max_xy = visible_pixels.max(axis=0)
        center = (min_xy + max_xy) * 0.5
        extent = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
        bbox_size = max(2.0, extent * 1.2)
    else:
        center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        bbox_size = float(max(width, height) * 0.5)

    # Equivalent to transform_keypoints(..., 224) followed by
    # normalize_keypoints_to_patch(...): coordinates are approximately [-1,1]
    # around the person crop center.
    patch = (pixels - center[None, :]) * (2.0 / bbox_size)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    patch = np.clip(patch, -4.0, 4.0)

    max_res = float(max(width, height))
    location = np.array(
        [
            2.0 * float(center[0]) / max_res - float(width) / max_res,
            2.0 * float(center[1]) / max_res - float(height) / max_res,
            bbox_size / max_res,
        ],
        dtype=np.float32,
    )
    feature = np.concatenate((patch.reshape(-1), location), axis=0).astype(np.float32)
    mask = (scores < 0.3).astype(np.bool_)
    return feature, mask, location


def _canonicalize_wham_coordinates(predicted: np.ndarray) -> np.ndarray:
    """Convert WHAM camera-space joints into TexMotion's SMPL-X frame.

    The released WHAM motion encoder emits root-relative camera coordinates:
    image down is ``+Y`` and camera depth away from the camera is ``+Z``.  The
    rest of TexMotion consumes SMPL-X coordinates where up is ``+Y`` and
    forward is ``+Z``.  Keep this conversion at the adapter boundary so every
    downstream consumer (overlay, kinematics, and exported motion) sees one
    unambiguous coordinate contract.  A copy is returned because checkpoint
    tensors may share storage with the inference graph.
    """
    arr = np.asarray(predicted, dtype=np.float32).copy()
    if arr.ndim < 2 or arr.shape[-1] < 3:
        return arr
    arr[..., 1] *= -1.0
    arr[..., 2] *= -1.0
    return arr


def _identity_pose6d(n_joints: int = 24) -> np.ndarray:
    pose = np.zeros((n_joints, 6), dtype=np.float32)
    pose[:, 0] = 1.0
    pose[:, 4] = 1.0
    return pose


def _axis_angle_to_matrix_numpy(axis_angle: np.ndarray) -> np.ndarray:
    """Convert (..., 3) axis-angle values without a torch dependency."""
    values = np.asarray(axis_angle, dtype=np.float32)
    theta = np.linalg.norm(values, axis=-1, keepdims=True)
    axis = values / np.maximum(theta, 1e-8)
    x, y, z = np.moveaxis(axis, -1, 0)
    c = np.cos(theta[..., 0])
    s = np.sin(theta[..., 0])
    one = 1.0 - c
    matrix = np.empty(values.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = c + x * x * one
    matrix[..., 0, 1] = x * y * one - z * s
    matrix[..., 0, 2] = x * z * one + y * s
    matrix[..., 1, 0] = y * x * one + z * s
    matrix[..., 1, 1] = c + y * y * one
    matrix[..., 1, 2] = y * z * one - x * s
    matrix[..., 2, 0] = z * x * one - y * s
    matrix[..., 2, 1] = z * y * one + x * s
    matrix[..., 2, 2] = c + z * z * one
    small = theta[..., 0] < 1e-7
    if np.any(small):
        matrix[small] = np.eye(3, dtype=np.float32)
    return matrix


def _rotation_to_matrix_numpy(pose: Any) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float32)
    if values.shape[-2:] == (3, 3):
        return values
    if values.shape[-1] == 6:
        values = values.reshape(values.shape[:-1] + (3, 2))
        first = values[..., :, 0]
        second = values[..., :, 1]
        first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
        second = second - (first * second).sum(axis=-1, keepdims=True) * first
        second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
        third = np.cross(first, second)
        return np.stack((first, second, third), axis=-1)
    if values.shape[-1] == 3:
        return _axis_angle_to_matrix_numpy(values)
    if values.size == 24 * 6:
        return _rotation_to_matrix_numpy(values.reshape(24, 6))
    if values.size == 24 * 3:
        return _rotation_to_matrix_numpy(values.reshape(24, 3))
    raise ValueError(
        "SMPL initialization pose must be (24,3,3), (24,6), or (24,3); "
        f"received shape {values.shape}"
    )


def _matrix_to_rotation_6d_numpy(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    return values[..., :2, :].reshape(values.shape[:-2] + (6,)).astype(np.float32)


def _coerce_smpl_initialization(
    source: Any,
    n_joints: int = 17,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Normalize a configured SMPL seed to WHAM's two recurrent inputs.

    The official ``CustomDataset`` receives a first-frame HMR2/SMPL estimate
    and emits ``init_kp`` (root-centered J17 + normalized 2D later) and
    ``init_smpl`` (24 full-pose rotations in 6D).  This helper accepts either
    those already-decoded fields or the common ``global_orient``/``body_pose``
    representation so callers can provide an offline fixture/body-model
    result without importing WHAM's dataset package.
    """
    if source is None:
        return None
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        data = {
            name: getattr(source, name)
            for name in ("joints", "joints3d", "full_pose", "pose", "global_orient", "body_pose")
            if hasattr(source, name)
        }
    joints_value = (
        data.get("joints")
        if data.get("joints") is not None
        else data.get("joints3d", data.get("init_kp3d"))
    )
    if joints_value is None:
        raise ValueError("SMPL initialization requires joints/joints3d/init_kp3d")
    joints = np.asarray(
        joints_value.detach().cpu().numpy() if hasattr(joints_value, "detach") else joints_value,
        dtype=np.float32,
    )
    while joints.ndim > 2 and joints.shape[0] == 1:
        joints = joints[0]
    if joints.ndim != 2 or joints.shape[-1] < 3 or joints.shape[0] < n_joints:
        raise ValueError(f"SMPL initialization joints must contain ({n_joints},3), got {joints.shape}")
    joints = joints[:n_joints, :3].copy()
    if n_joints >= 13:
        joints -= joints[[11, 12]].mean(axis=0, keepdims=True)
    else:
        joints -= joints[:1]

    full_pose_value = data.get("full_pose", data.get("pose"))
    if full_pose_value is None and data.get("global_orient") is not None:
        global_orient = np.asarray(
            data["global_orient"].detach().cpu().numpy()
            if hasattr(data["global_orient"], "detach")
            else data["global_orient"],
            dtype=np.float32,
        )
        body_pose = np.asarray(
            data.get("body_pose", np.zeros((23, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        while global_orient.ndim > 2 and global_orient.shape[0] == 1:
            global_orient = global_orient[0]
        while body_pose.ndim > 2 and body_pose.shape[0] == 1:
            body_pose = body_pose[0]
        full_pose_value = np.concatenate(
            (global_orient.reshape(1, -1), body_pose.reshape(23, -1)), axis=0
        )
    if full_pose_value is None:
        full_pose = np.tile(np.eye(3, dtype=np.float32), (24, 1, 1))
    else:
        pose_values = (
            full_pose_value.detach().cpu().numpy()
            if hasattr(full_pose_value, "detach")
            else full_pose_value
        )
        pose_values = np.asarray(pose_values, dtype=np.float32)
        while pose_values.ndim > 3 and pose_values.shape[0] == 1:
            pose_values = pose_values[0]
        if pose_values.shape == (24, 6) or pose_values.shape == (24, 3):
            full_pose = _rotation_to_matrix_numpy(pose_values)
        elif pose_values.shape == (24, 3, 3):
            full_pose = pose_values
        elif pose_values.size in (24 * 6, 24 * 3):
            full_pose = _rotation_to_matrix_numpy(pose_values.reshape(24, -1))
        else:
            raise ValueError(f"SMPL initialization full pose has unsupported shape {pose_values.shape}")
    return joints.astype(np.float32), _matrix_to_rotation_6d_numpy(full_pose), "configured"


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)
    x, y, z, w = np.moveaxis(q, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _matrix_to_axis_angle_numpy(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    cosine = np.clip((np.trace(values, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cosine)
    vector = np.stack(
        (values[..., 2, 1] - values[..., 1, 2],
         values[..., 0, 2] - values[..., 2, 0],
         values[..., 1, 0] - values[..., 0, 1]),
        axis=-1,
    ) * 0.5
    denominator = np.sin(angle)[..., None]
    axis_angle = vector * (angle[..., None] / np.maximum(denominator, 1e-7))
    axis_angle = np.nan_to_num(axis_angle, nan=0.0, posinf=0.0, neginf=0.0)
    small = angle < 1e-6
    if np.any(small):
        axis_angle[small] = 0.0
    return axis_angle.astype(np.float32)


def _camera_angular_velocity_from_poses(poses: Any, fps: float = 30.0) -> np.ndarray:
    """Port ``convert_dpvo_to_cam_angvel`` without DPVO/PyTorch imports."""
    values = np.asarray(poses, dtype=np.float32)
    if values.ndim != 2 or values.shape[-1] < 7:
        raise ValueError("camera poses must have shape (frames, 7) [txyz + qxyzw]")
    n_frames = values.shape[0]
    if n_frames <= 1:
        return np.zeros((n_frames, 6), dtype=np.float32)
    world2cam = _quaternion_xyzw_to_matrix(values[:, 3:7])
    camera_to_world = np.swapaxes(world2cam, -1, -2)
    relative = camera_to_world[:-1] @ np.swapaxes(camera_to_world[1:], -1, -2)
    axis_angle = _matrix_to_axis_angle_numpy(relative)
    relative_rot = _axis_angle_to_matrix_numpy(axis_angle)
    angular_velocity = _matrix_to_rotation_6d_numpy(relative_rot)
    angular_velocity -= np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    angular_velocity *= float(fps)
    return np.concatenate((angular_velocity, angular_velocity[:1]), axis=0).astype(np.float32)


def _load_local_asset_payload(source: Any, label: str) -> Any:
    """Read a local WHAM companion archive without importing research code.

    Camera calibration files and DPVO exports are deliberately separate
    artifacts.  This helper only decodes the common container formats; the
    caller decides whether the payload is a trajectory, calibration mapping,
    or a model checkpoint.
    """
    if not isinstance(source, (str, os.PathLike)):
        return source
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"WHAM {label} asset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        archive = np.load(path)
        # Keep the archive mapping so callers can distinguish camera poses
        # from calibration/other arrays instead of guessing the first key.
        return {key: archive[key] for key in archive.files}
    if suffix in (".pt", ".pth", ".tar"):
        if torch is None:
            raise RuntimeError(f"PyTorch is required to read a WHAM {label} asset")
        return _safe_torch_load(torch, str(path), "cpu")
    if suffix in (".json", ".yaml", ".yml"):
        try:
            if suffix == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            try:
                import yaml
            except ImportError:
                # Calibration files normally contain scalar fx/fy/cx/cy
                # fields.  Keep this tiny fallback so the lightweight profile
                # can use those files without making PyYAML mandatory.
                values: Dict[str, Any] = {}
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.split("#", 1)[0].strip()
                    if not line or ":" not in line:
                        continue
                    key, raw = line.split(":", 1)
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        value: Any = float(raw)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = raw.strip("'\"")
                    values[key.strip()] = value
                return values
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"could not read WHAM {label} asset: {path}: {exc}") from exc
    raise ValueError(f"unsupported WHAM {label} asset: {path.suffix}")


def _camera_motion_value(source: Any) -> Any:
    """Extract a trajectory/angular-velocity array from a structured payload."""
    if isinstance(source, Mapping):
        for key in (
            "poses", "cameraPoses", "camera_poses", "slamResults", "slam_results",
            "angular_velocity", "cameraAngularVelocity", "camera_angular_velocity",
            "motion", "trajectory", "camera_motion",
        ):
            if key in source and source[key] is not None:
                return source[key]
        return None
    return source


def _is_tensor_mapping(value: Any) -> bool:
    """Return whether a mapping looks like a PyTorch state dictionary."""
    return bool(
        isinstance(value, Mapping)
        and value
        and all(isinstance(key, str) for key in value)
        and any(hasattr(item, "shape") or hasattr(item, "detach") for item in value.values())
    )


def _looks_like_model_checkpoint(value: Any) -> bool:
    """Distinguish raw model checkpoints from explicit feature archives.

    ``torch.save`` is used for both WHAM frame-feature archives and model
    checkpoints, so the extension alone is not provenance.  Recognize the
    standard state-dict wrappers before coercing a path into an ``(N, D)``
    feature matrix.  A mapping with an explicit ``features`` field remains an
    archive even if it carries unrelated metadata.
    """
    if not isinstance(value, Mapping):
        return False
    if any(key in value for key in ("features", "image_features", "imageFeatures")):
        return False
    for key in (
        "state_dict",
        "model_state_dict",
        "modelStateDict",
        "weights",
        "params",
        "model",
        "network",
        "module",
        "backbone",
    ):
        candidate = value.get(key)
        if _is_tensor_mapping(candidate):
            return True
    # A bare state dict is also a valid torch checkpoint.  Require tensor-like
    # values and avoid treating arbitrary scalar metadata mappings as models.
    return _is_tensor_mapping(value)


def _is_camera_calibration_payload(source: Any) -> bool:
    """Return true for an intrinsics/config mapping without motion rows."""
    if not isinstance(source, Mapping):
        return False
    keys = {str(key).strip().lower() for key in source.keys()}
    return bool(
        {"fx", "fy", "cx", "cy"}.issubset(keys)
        or "intrinsics" in keys
        or "camera_matrix" in keys
        or "k" in keys
    )


def _coerce_camera_angular_velocity(source: Any, frame_count: int, fps: float) -> Optional[np.ndarray]:
    if source is None:
        return None
    if isinstance(source, (str, os.PathLike)):
        source = _load_local_asset_payload(source, "camera/DPVO")
    source = _camera_motion_value(source)
    if source is None:
        raise ValueError("camera asset contains no DPVO poses or WHAM angular velocity")
    values = np.asarray(
        source.detach().cpu().numpy() if hasattr(source, "detach") else source,
        dtype=np.float32,
    )
    if values.ndim == 2 and values.shape[-1] == 7:
        values = _camera_angular_velocity_from_poses(values, fps)
    elif values.ndim == 2 and values.shape[-1] == 6:
        if values.shape[0] == frame_count - 1 and frame_count:
            values = np.concatenate((values, values[:1]), axis=0)
        elif values.shape[0] != frame_count:
            raise ValueError(
                f"camera angular velocity must contain {frame_count} frames, got {values.shape[0]}"
            )
    else:
        raise ValueError("camera motion must be DPVO (N,7) poses or WHAM (N,6) angular velocity")
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _coerce_image_features(
    source: Any,
    frame_count: int,
    expected_dim: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    *,
    expected_fps: Optional[float] = None,
    expected_source_fps: Optional[float] = None,
    expected_timestamps: Optional[Sequence[float]] = None,
    expected_trim_start: Optional[float] = None,
    expected_trim_end: Optional[float] = None,
    expected_video_hash: Optional[str] = None,
    expected_video_path: Optional[str | os.PathLike[str]] = None,
    require_archive_metadata: bool = False,
) -> np.ndarray:
    """Load and verify a frame-aligned image-feature sequence.

    A feature archive may carry ``frame_indices`` metadata.  If it does, the
    metadata is authoritative for row identity and must match the requested
    source order exactly; silently retiming a trimmed clip is worse than
    rejecting the archive.  Plain arrays remain supported and are interpreted
    in the caller-provided order.
    """

    expected_indices: Optional[List[int]] = None

    def parse_indices(values: Any, label: str) -> List[int]:
        try:
            raw_values = np.asarray(values).reshape(-1).tolist()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} are not readable") from exc
        parsed: List[int] = []
        for value in raw_values:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{label} are not integer-like")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{label} are not integer-like") from exc
            if not np.isfinite(numeric) or numeric != math.trunc(numeric):
                raise ValueError(f"{label} are not integer-like")
            parsed.append(int(numeric))
        return parsed

    if frame_indices is not None:
        expected_indices = parse_indices(frame_indices, "frame_indices")
        if len(expected_indices) != frame_count:
            raise ValueError(
                f"frame_indices must have length {frame_count}, got {len(expected_indices)}"
            )
        if len(set(expected_indices)) != len(expected_indices):
            raise ValueError("frame_indices contain duplicate source frame IDs")

    embedded_indices: Any = None
    archive_metadata: Dict[str, Any] = {}

    def metadata_mapping(value: Any) -> Dict[str, Any]:
        """Decode optional JSON/NumPy metadata without changing feature rows."""

        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return {}
            value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return {}
            return metadata_mapping(parsed)
        return {}

    def metadata_value(*names: str) -> Any:
        for name in names:
            if name in archive_metadata and archive_metadata[name] is not None:
                value = archive_metadata[name]
                if isinstance(value, np.ndarray) and value.size == 1:
                    return value.reshape(-1)[0].item()
                return value
        return None

    def as_float(value: Any, label: str) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"feature archive {label} is not numeric") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"feature archive {label} is not finite")
        return parsed

    def compare_float(actual: Any, expected: Optional[float], label: str) -> None:
        if expected is None or actual is None:
            return
        actual_value = as_float(actual, label)
        expected_value = as_float(expected, label)
        if actual_value is None or expected_value is None:
            return
        tolerance = max(1.0e-4, 1.0e-4 * max(abs(actual_value), abs(expected_value), 1.0))
        if abs(actual_value - expected_value) > tolerance:
            raise ValueError(
                f"feature archive {label} does not match requested sequence: "
                f"archive={actual_value}, requested={expected_value}"
            )

    def file_sha256(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            path = Path(str(value)).expanduser()
        except (TypeError, ValueError, OSError):
            return None
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
        return digest.hexdigest()

    def verify_archive_metadata() -> None:
        """Reject stale trim/FPS/video archives before the Integrator sees them."""

        archive_fps = metadata_value("fps", "output_fps", "target_fps", "targetFps")
        archive_source_fps = metadata_value(
            "source_fps", "sourceFps", "original_fps", "originalFps"
        )
        archive_timestamps = metadata_value("timestamps", "timestamp_sec", "timestampsSec")
        archive_trim_start = metadata_value(
            "trim_start", "trimStart", "trim_start_sec", "trimStartSec"
        )
        archive_trim_end = metadata_value(
            "trim_end", "trimEnd", "trim_end_sec", "trimEndSec"
        )
        archive_video_hash = metadata_value(
            "video_hash", "videoHash", "source_video_hash", "sourceVideoHash"
        )
        archive_video_path = metadata_value(
            "source_video", "sourceVideo", "video_path", "videoPath"
        )

        archive_has_identity = any(
            value is not None
            for value in (
                archive_fps,
                archive_source_fps,
                archive_timestamps,
                archive_trim_start,
                archive_trim_end,
                archive_video_hash,
                archive_video_path,
            )
        )
        if require_archive_metadata and not archive_has_identity:
            raise ValueError(
                "feature archive is missing FPS/trim/video identity metadata"
            )

        compare_float(archive_fps, expected_fps, "fps")
        compare_float(archive_source_fps, expected_source_fps, "source_fps")
        compare_float(archive_trim_start, expected_trim_start, "trim_start")
        compare_float(archive_trim_end, expected_trim_end, "trim_end")

        if archive_timestamps is not None and expected_timestamps is not None:
            try:
                actual_times = np.asarray(archive_timestamps, dtype=np.float64).reshape(-1)
                requested_times = np.asarray(expected_timestamps, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError("feature archive timestamps are not numeric") from exc
            if actual_times.shape != requested_times.shape or not np.isfinite(actual_times).all():
                raise ValueError(
                    "feature archive timestamps are not aligned with requested sequence"
                )
            if not np.allclose(actual_times, requested_times, rtol=1.0e-4, atol=1.0e-4):
                raise ValueError(
                    "feature archive timestamps do not match requested trim/frame sequence"
                )

        if expected_video_hash is not None and archive_video_hash is not None:
            if str(archive_video_hash).strip().lower() != str(expected_video_hash).strip().lower():
                raise ValueError("feature archive video hash does not match requested video")
        if expected_video_path is not None:
            requested_hash = file_sha256(expected_video_path)
            if requested_hash is not None and archive_video_hash is not None:
                if str(archive_video_hash).strip().lower() != requested_hash.lower():
                    raise ValueError("feature archive video hash does not match requested video")
            elif archive_video_path is not None:
                try:
                    requested_path = Path(str(expected_video_path)).expanduser().resolve()
                    archive_path = Path(str(archive_video_path)).expanduser().resolve()
                except (TypeError, ValueError, OSError):
                    requested_path = archive_path = None
                if requested_path is not None and archive_path is not None and requested_path != archive_path:
                    raise ValueError(
                        "feature archive source video does not match requested video"
                    )

    def unpack_mapping(value: Mapping[str, Any]) -> Any:
        nonlocal embedded_indices, archive_metadata
        nested_metadata = metadata_mapping(value.get("metadata"))
        archive_metadata = {**nested_metadata, **{str(key): item for key, item in value.items()}}
        embedded_indices = value.get(
            "frame_indices",
            value.get("frameIndices", value.get("source_frame_indices", value.get("sourceFrameIndices"))),
        )
        if _looks_like_model_checkpoint(value):
            raise _UnverifiedFeatureSourceError(
                "WHAM image feature source is a raw model checkpoint/state dict, "
                "not a verified per-frame feature archive"
            )
        return value.get("features", value.get("image_features", value.get("imageFeatures")))

    if callable(source):
        source = source(frame_count)
    if isinstance(source, Mapping):
        source = unpack_mapping(source)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"WHAM image feature archive not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            source = np.load(path)
        elif suffix == ".npz":
            archive = np.load(path)
            key = "features" if "features" in archive else archive.files[0]
            if "metadata" in archive:
                archive_metadata = metadata_mapping(archive["metadata"])
            for metadata_key in (
                "fps", "source_fps", "trim_start", "trim_end", "source_video",
                "video_hash", "timestamps", "frame_indices",
            ):
                if metadata_key in archive:
                    archive_metadata[metadata_key] = archive[metadata_key]
            if "frame_indices" in archive:
                embedded_indices = archive["frame_indices"]
            source = archive[key]
        elif suffix in (".pt", ".pth", ".tar"):
            if torch is None:
                raise RuntimeError("PyTorch is required to read a WHAM image feature archive")
            source = _safe_torch_load(torch, str(path), "cpu")
            if isinstance(source, Mapping):
                source = unpack_mapping(source)
        else:
            raise ValueError(f"unsupported WHAM image feature archive: {path.suffix}")
    if embedded_indices is None:
        embedded_indices = metadata_value(
            "frame_indices", "frameIndices", "source_frame_indices", "sourceFrameIndices"
        )
    if source is None:
        raise ValueError("WHAM image feature source is empty")
    raw_source = source.detach().cpu().numpy() if hasattr(source, "detach") else source
    raw_values = np.asarray(raw_source)
    if not np.issubdtype(raw_values.dtype, np.floating):
        raise ValueError(
            "WHAM image features must use a floating-point dtype (float32/float64), "
            f"got {raw_values.dtype}"
        )
    values = np.asarray(raw_values, dtype=np.float32)
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(
            f"WHAM image features must have 2 dimensions (N, D), got {values.shape}"
        )
    is_single_frame_probe = frame_count == 1 and values.shape[0] > 1
    if values.shape[0] != frame_count:
        if is_single_frame_probe:
            values = values[:1]
        else:
            raise ValueError(
                f"WHAM image features must have shape ({frame_count}, D), got {values.shape}"
            )
    if expected_dim is not None and values.shape[1] != expected_dim:
        raise ValueError(
            f"WHAM image features require dimension {expected_dim}, got {values.shape[1]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("WHAM image features contain non-finite values")
    if embedded_indices is not None:
        archive_indices = parse_indices(embedded_indices, "feature archive frame_indices")
        if is_single_frame_probe:
            if len(archive_indices) == 0:
                raise ValueError("feature archive frame_indices is empty")
        else:
            if len(archive_indices) != frame_count:
                raise ValueError(
                    f"feature archive frame_indices must have length {frame_count}, got {len(archive_indices)}"
                )
            if len(set(archive_indices)) != len(archive_indices):
                raise ValueError("feature archive frame_indices contain duplicates")
            if expected_indices is not None and archive_indices != expected_indices:
                raise ValueError(
                    "feature archive frame_indices are not aligned with requested frame_indices"
                )
    if not is_single_frame_probe:
        verify_archive_metadata()
    return values


def _infer_architecture(state: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Infer WHAM dimensions from checkpoint tensors before model creation."""
    result: Dict[str, Any] = {
        "n_joints": int(config.get("n_joints", 17)),
        "in_dim": int(config.get("in_dim", 37)),
        "d_embed": int(config.get("d_embed", 512)),
        "n_layers": int(config.get("n_layers", 3)),
        "rnn_type": str(config.get("rnn_type", config.get("layer", "LSTM"))),
        "d_feat": int(config.get("d_feat", 1024)),
    }
    embed_weight = state.get("motion_encoder.embed_layer.weight")
    if hasattr(embed_weight, "shape") and len(embed_weight.shape) == 2:
        result["d_embed"] = int(embed_weight.shape[0])
        result["in_dim"] = int(embed_weight.shape[1])
        result["n_joints"] = max(1, int((result["in_dim"] - 3) // 2))
    layer_ids = []
    for key in state:
        prefix = "motion_encoder.regressor.rnn.weight_ih_l"
        if key.startswith(prefix):
            try:
                layer_ids.append(int(key[len(prefix):].split(".", 1)[0]))
            except ValueError:
                pass
    if layer_ids:
        result["n_layers"] = max(layer_ids) + 1
    integrator_weight = state.get("integrator.layer1.weight")
    if hasattr(integrator_weight, "shape") and len(integrator_weight.shape) == 2:
        context_dim = result["d_embed"] + result["n_joints"] * 3
        inferred = int(integrator_weight.shape[1]) - context_dim
        if inferred <= 0:
            raise ValueError(
                "WHAM Integrator checkpoint tensor has no positive image-feature input dimension: "
                f"inferred D={inferred}"
            )
        configured = config.get("d_feat")
        if configured is not None and int(configured) != inferred:
            raise ValueError(
                "WHAM Integrator feature dimension does not match configuration: "
                f"checkpoint requires D={inferred}, configured d_feat={int(configured)}"
            )
        # The checkpoint is authoritative; a config value is only a
        # consistency assertion, never a way to reinterpret the weight
        # matrix and construct a wrong Integrator.
        result["d_feat"] = inferred
    return result


def _extract_embedded_smpl_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    prefix = "smpl."
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix) and hasattr(value, "shape")
    }


def _first_config_value(config: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return None


def _config_bool(config: Mapping[str, Any], *names: str, default: bool = False) -> bool:
    """Read editor/JSON booleans without treating ``"false"`` as true."""

    value = _first_config_value(config, *names)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off", "none", "null"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _load_configured_symbol(value: Any, label: str) -> Any:
    """Resolve a callable/module configured as a string or Python object.

    Quality integrations are intentionally configured at runtime so importing
    the normal MediaPipe profile never imports PyTorch or a research checkout.
    Accept both ``package.module:attribute`` and a Python file path because the
    editor stores optional model code beside downloaded assets.
    """
    if value is None or not isinstance(value, (str, os.PathLike)):
        return value
    reference = os.path.expanduser(str(value))
    module_name = reference
    attribute_name: Optional[str] = None
    drive, _ = os.path.splitdrive(reference)
    if ":" in reference and not os.path.isfile(reference):
        # Use the final colon so ``C:\\...\\model.py:create_model`` keeps its
        # drive prefix intact.  A module:attribute reference is still accepted
        # when it is not a Windows drive path; for a missing Windows file,
        # retain the complete path so the resulting diagnostic names it.
        candidate_name, separator, candidate_attribute = reference.rpartition(":")
        if separator and (os.path.isfile(candidate_name) or not drive):
            module_name = candidate_name
            attribute_name = candidate_attribute
    if os.path.isfile(module_name):
        path = Path(module_name).resolve()
        module_name = f"texmotion_inline_{path.stem}_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load {label} module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    if attribute_name:
        try:
            return getattr(module, attribute_name)
        except AttributeError as exc:
            raise AttributeError(
                f"{label} {attribute_name!r} is not defined by {module_name!r}"
            ) from exc
    return module


def _runner_like(value: Any) -> bool:
    """Return whether ``value`` already implements the feature runner seam."""
    return callable(getattr(value, "extract_sequence", None)) or callable(value)


def _sequence_cache_identity(
    frames_bgr: Sequence[np.ndarray],
    timestamps_sec: Sequence[float],
    frame_indices: Sequence[int],
) -> str:
    """Build a compact identity for generated per-video camera caches.

    A preflight smoke pass can have the same length, frame index, and timestamp
    as a real one-frame clip while containing different pixels.  Shape-only
    cache checks therefore reuse stale camera motion.  Include source identity
    and a bounded pixel sample so cache invalidation remains cheap for long
    clips but distinguishes the black smoke frame from the requested video.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(len(frames_bgr)).encode("utf-8"))
    for frame, timestamp, frame_index in zip(frames_bgr, timestamps_sec, frame_indices):
        digest.update(f"|{int(frame_index)}:{float(timestamp):.9f}".encode("utf-8"))
        if frame is None:
            digest.update(b"|none")
            continue
        array = np.asarray(frame)
        digest.update(f"|{array.shape}:{array.dtype}".encode("utf-8"))
        if array.size:
            flat = np.ascontiguousarray(array).reshape(-1)
            stride = max(1, int(flat.size // 4096))
            digest.update(np.ascontiguousarray(flat[::stride]).tobytes())
    return digest.hexdigest()


_SHA256_CACHE: Dict[str, Tuple[float, int, str]] = {}
_SHA256_LOCK = threading.Lock()


def _sha256_file(path: Any) -> Optional[str]:
    """Hash an optional local asset for runner/integrator provenance."""

    if path is None:
        return None
    try:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            return None
        stat = candidate.stat()
        cache_key = os.path.normcase(str(candidate))
        with _SHA256_LOCK:
            cached = _SHA256_CACHE.get(cache_key)
            if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                return cached[2]
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hexdigest = digest.hexdigest()
        with _SHA256_LOCK:
            _SHA256_CACHE[cache_key] = (stat.st_mtime, stat.st_size, hexdigest)
        return hexdigest
    except OSError:
        return None


class NativeWHAMAdapter:
    """Run WHAM's temporal 3D lifting core with local TexMotion inputs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.model_path = self.config.get("modelPath") or self.config.get("model_path")
        self.model_directory = self.config.get("modelDirectory") or self.config.get("model_directory")
        self.device = _device_name(self.config)
        self._torch = self.config.get("torch") or torch
        self._model: Any = None
        self._detector: Any = self.config.get("detector")
        self._error: Optional[str] = None
        self._preflight_status = "not_run"
        self._preflight_error: Optional[str] = None
        self._loaded_parameter_count = 0
        self._detector_name = "mediapipe"
        self._architecture: Dict[str, Any] = {}
        self._loaded_components: Dict[str, bool] = {
            "nativeMotionEncoder": False,
            "imageFeatureIntegrator": False,
            "trajectoryDecoder": False,
            "smplDecoder": False,
            "trajectoryRefiner": False,
        }
        self._smpl_model: Any = None
        self._smpl_seed: Optional[Tuple[np.ndarray, np.ndarray, str]] = None
        # An explicit caller supplied initialization must remain authoritative.
        # The automatic MediaPipe seed is allowed to replace only the neutral
        # checkpoint seed, which otherwise has no image/camera orientation.
        self._smpl_seed_explicit = False
        self._smpl_error: Optional[str] = None
        self._optional_errors: List[str] = []
        self._runtime_warnings: List[str] = []
        self._preprocess_directory = _first_config_value(
            self.config,
            "whamPreprocessDirectory",
            "wham_preprocess_directory",
            "preprocessOutputDirectory",
            "preprocess_output_directory",
        )
        if self._preprocess_directory is not None:
            self._preprocess_directory = str(self._preprocess_directory)
        self._preprocess_manifest_path: Optional[str] = None
        self._auto_preprocess_used = False
        self._auto_feature_preprocess_used = False
        self._auto_camera_preprocess_used = False
        self._feature_preprocess_runner_name: Optional[str] = None
        self._camera_preprocess_runner_name: Optional[str] = None
        # ``imageFeatureBackbonePath`` is a ViTPose model checkpoint.  It is
        # not a frame-feature archive and must never be passed to
        # ``_coerce_image_features`` as if it contained an (N,D) tensor.
        self._feature_backbone_path = _first_config_value(
            self.config,
            "imageFeatureBackbonePath",
            "image_feature_backbone_path",
        )
        feature_candidate = _first_config_value(
            self.config,
            "imageFeatures",
            "image_features",
            "imageFeaturesPath",
            "image_features_path",
            "imageFeatureArchive",
            "image_feature_archive",
            "imageFeaturePath",
            "image_feature_path",
        )
        if self._feature_backbone_path is not None:
            self._feature_backbone_path = str(self._feature_backbone_path)
        if isinstance(feature_candidate, (str, os.PathLike)):
            feature_suffix = Path(feature_candidate).suffix.lower()
            if feature_suffix not in (".npy", ".npz", ".pt", ".pth", ".tar"):
                # Asset manifests commonly call an HMR2/ViT checkpoint an
                # imageFeatureArchive.  It is a backbone, not a precomputed
                # frame-feature sequence; do not attempt to unpickle it as
                # features or hide the missing extractor behind a fallback.
                self._feature_backbone_path = str(feature_candidate)
                feature_candidate = None
        self._feature_source = feature_candidate
        # A precomputed archive or caller-supplied extractor is an explicit
        # input.  The bundled local descriptor is intentionally *not* an
        # acceptable substitute for learned WHAM image features; keep source
        # provenance separate so a stale/generated cache cannot look verified.
        self._feature_source_provenance = (
            "verified_archive" if feature_candidate is not None else "none"
        )
        self._feature_source_verified = feature_candidate is not None
        # Inline ViTPose is optional and resolved lazily after the WHAM
        # checkpoint has revealed its expected feature width.  Keep the
        # existing ``imageFeatureExtractor`` contract separate: callers that
        # provide one still take precedence over this convenience runner.
        self._feature_runner: Any = None
        self._feature_runner_resolution_attempted = False
        self._feature_runner_error: Optional[str] = None
        self._feature_runner_module: Optional[str] = None
        self._feature_runner_checkpoint: Optional[str] = None
        self._feature_runner_model_definition: Any = None
        self._feature_runner_config_path: Any = None
        self._feature_runner_active = False
        self._feature_runner_kind: Optional[str] = None
        self._feature_error_code: Optional[str] = None
        self._feature_next_action: Optional[str] = None
        # Compatibility aliases retained for diagnostics consumers.  The
        # fallback is active as a labeled decision, but ``Consumed`` remains
        # false because these descriptors never enter WHAM's learned path.
        self._feature_runner_metadata: Dict[str, Any] = {}
        self._feature_fallback_active = False
        self._feature_fallback_kind: Optional[str] = None
        self._feature_fallback_reason: Optional[str] = None
        self._feature_fallback_consumed = False
        self._feature_adopted_frame_indices: List[int] = []
        self._feature_input_frame_indices: List[int] = []
        self._requested_backend = str(
            _first_config_value(self.config, "requestedBackend", "requested_backend", "backend")
            or "wham"
        )
        self._allow_mediapipe_fallback = _config_bool(
            self.config,
            "AllowMediaPipeFallback",
            "allowMediaPipeFallback",
            "allow_mediapipe_fallback",
            default=False,
        )
        # Motion arrays can be supplied directly, through a precomputed
        # camera/SLAM archive, or by a caller-provided extractor.  Camera
        # intrinsics and DPVO weights are separate assets and are retained for
        # diagnostics without being mistaken for frame-wise motion.
        self._camera_source = _first_config_value(
            self.config,
            "cameraAngularVelocity",
            "camera_angular_velocity",
            "cameraPoses",
            "camera_poses",
            "slamResults",
            "slam_results",
            "cameraMotion",
            "camera_motion",
            "cameraMotionPath",
            "camera_motion_path",
            "slamResultsPath",
            "slam_results_path",
        )
        self._camera_asset_path = _first_config_value(
            self.config,
            "cameraModelPath",
            "camera_model_path",
            "cameraPath",
            "camera_path",
        )
        self._dpvo_asset_path = _first_config_value(
            self.config,
            "dpvoModelPath",
            "dpvo_model_path",
            "dpvoPath",
            "dpvo_path",
        )
        if self._camera_asset_path is not None:
            self._camera_asset_path = str(self._camera_asset_path)
        if self._dpvo_asset_path is not None:
            self._dpvo_asset_path = str(self._dpvo_asset_path)
        # Preserve the long-standing private/source alias for camera files so
        # external adapters that inspect it continue to work.  The loader
        # below distinguishes motion archives from calibration-only payloads.
        if self._camera_source is None and self._camera_asset_path:
            self._camera_source = self._camera_asset_path
        self._camera_calibration: Optional[Mapping[str, Any]] = None
        self._dpvo_weights_available = False
        self._camera_angular_velocity: Optional[np.ndarray] = None
        self._camera_preprocess_signature: Optional[str] = None
        self._camera_motion_source = "none"
        self._camera_motion_provenance: Dict[str, Any] = {
            "source": "none",
            "verified": False,
            "runner": None,
            "asset": None,
        }
        self._dpvo_verified = False
        self._incompatible_assets: List[str] = []
        self._smpl_output_used = False
        self._image_features_used = False
        self._trajectory_used = False
        self._trajectory_refiner_used = False
        self._full_decoder_requested = bool(
            self.config.get("enableOfficialDecoders", self.config.get("enable_decoders", True))
        )
        self._allow_neutral_seed = bool(
            self.config.get("allowNeutralSMPLInitialization", self.config.get("allow_neutral_smpl", True))
        )
        self._strict_optional_assets = bool(
            self.config.get("strictOptionalAssets", self.config.get("strict_optional_assets", False))
        )
        self._require_image_features = bool(
            self.config.get("requireImageFeatures", self.config.get("require_image_features", False))
        )
        self._require_smpl = bool(
            self.config.get("requireSMPLInitialization", self.config.get("require_smpl_initialization", False))
        )
        self._enable_trajectory = bool(
            self.config.get("enableTrajectory", self.config.get("enable_trajectory", True))
        )
        self._enable_trajectory_refiner = bool(
            self.config.get("enableTrajectoryRefiner", self.config.get("enable_trajectory_refiner", False))
        )
        self._fps = float(self.config.get("fps", self.config.get("frameRate", 30.0)) or 30.0)

    @property
    def error(self) -> Optional[str]:
        return self._error

    def is_available(self) -> bool:
        return bool(
            self._torch is not None
            and self.model_path
            and os.path.isfile(os.path.abspath(os.path.expanduser(str(self.model_path))))
        )

    def _resolve_model_path(self) -> str:
        if not self.model_path:
            raise FileNotFoundError("WHAM checkpoint path is empty")
        path = os.path.abspath(os.path.expanduser(str(self.model_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"WHAM checkpoint not found: {path}")
        return path

    def _initialize_detector(self) -> None:
        if self._detector is not None:
            initialize = getattr(self._detector, "initialize", None)
            if initialize is not None and initialize() is False:
                raise RuntimeError("WHAM input detector failed to initialize")
            self._detector_name = str(getattr(self._detector, "name", "custom") or "custom")
            return

        # WHAM consumes 2D keypoints.  Prefer a local RTMPose ONNX detector
        # when one is available: it is faster and more accurate around
        # occlusions, and does not require a MediaPipe task asset.  The quality
        # path is explicitly offline, so this probe never auto-downloads
        # weights.  MediaPipe remains the compatible fallback for installations
        # that only have a local ``pose_landmarker_*.task`` file.
        failures: List[str] = []
        try:
            module = importlib.import_module("pose_pipeline.backends.rtmpose_backend")
            detector_cls = getattr(module, "RTMPoseBackend")
            detector = detector_cls(
                model_directory=self.model_directory,
                allow_download=False,
            )
            if detector.is_available() and detector.initialize():
                self._detector = detector
                self._detector_name = "rtmpose"
                return
            failures.append("RTMPose ONNX model or ONNX Runtime is unavailable")
            close = getattr(detector, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            failures.append(f"RTMPose: {exc}")

        try:
            # Import the backend directly to avoid importing the quality
            # factory recursively while this adapter is being initialized.
            module = importlib.import_module("pose_pipeline.backends.mediapipe_backend")
            detector_cls = getattr(module, "MediaPipeBackend")
            detector = detector_cls(
                model_directory=self.model_directory,
                allow_download=False,
            )
            if detector.initialize():
                self._detector = detector
                self._detector_name = "mediapipe"
                return
            failures.append("MediaPipe task asset or legacy Solutions API is unavailable")
            close = getattr(detector, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            failures.append(f"MediaPipe: {exc}")

        self._detector = None
        detail = "; ".join(failures) if failures else "no detector candidates were found"
        raise RuntimeError(
            "WHAM input detector unavailable in offline mode: " + detail +
            ". Download a local RTMPose ONNX or MediaPipe pose_landmarker task asset."
        )

    def _initialize_smpl_assets(self, state: Mapping[str, Any]) -> None:
        """Resolve embedded/configured SMPL support without network access."""
        self._smpl_model = None
        self._smpl_seed = None
        self._smpl_error = None

        configured_seed = _first_config_value(
            self.config,
            "smplInitialization",
            "smpl_initialization",
            "initialSMPL",
            "initial_smpl",
        )
        if configured_seed is not None:
            self._smpl_seed_explicit = True
            try:
                self._smpl_seed = _coerce_smpl_initialization(configured_seed)
            except Exception as exc:
                self._smpl_error = f"configured SMPL initialization is invalid: {exc}"

        configured_model = _first_config_value(
            self.config,
            "smplModel",
            "smpl_model",
            "bodyModel",
            "body_model",
        )
        # Unity's asset resolver exposes the body-model path through both the
        # canonical ``*ModelPath`` keys and legacy ``bodyModel``/``smplModel``
        # aliases.  A path-valued alias is configuration data, not an
        # already-instantiated callable SMPL module.  Treating it as a module
        # makes the neutral-seed probe invoke a string and leaves the decoder
        # disabled with the misleading ``'str' object is not callable`` error.
        model_path = _first_config_value(
            self.config,
            "smplModelPath",
            "smpl_model_path",
            "smplxModelPath",
            "smplx_model_path",
            "bodyModelPath",
            "body_model_path",
        )
        if isinstance(configured_model, (str, os.PathLike)):
            if model_path is None:
                model_path = configured_model
            configured_model = None
        if configured_model is not None:
            self._smpl_model = configured_model

        # The official checkpoint carries all neutral SMPL tensors under
        # smpl.*.  Prefer those over a separate smplx installation so the
        # shipped 527 MB archive remains sufficient for CPU inference.
        if self._smpl_model is None and EmbeddedSMPL is not None:
            embedded = _extract_embedded_smpl_state(state)
            if embedded:
                try:
                    self._smpl_model = EmbeddedSMPL(embedded).to(self.device).eval()
                except Exception as exc:
                    self._smpl_error = f"embedded WHAM SMPL buffers are unusable: {exc}"

        if self._smpl_model is None and model_path:
            expanded = os.path.abspath(os.path.expanduser(str(model_path)))
            if not os.path.exists(expanded):
                self._smpl_error = f"SMPL body-model asset not found: {expanded}"
            else:
                # Import only when explicitly requested.  The compact native
                # path must remain usable in environments without smplx.
                try:
                    smplx = importlib.import_module("smplx")
                    self._smpl_model = smplx.create(
                        expanded,
                        model_type="smpl",
                        gender="neutral",
                        batch_size=1,
                        create_transl=False,
                    ).to(self.device).eval()
                except Exception as exc:
                    self._smpl_error = (
                        f"SMPL body-model asset could not be loaded from {expanded}: {exc}"
                    )

        if self._smpl_model is not None and self._smpl_seed is None:
            neutral = getattr(self._smpl_model, "neutral_initialization", None)
            if callable(neutral):
                try:
                    joints, pose = neutral(self._torch.device(self.device))
                    coerced = _coerce_smpl_initialization(
                        {
                            "joints": joints.detach().cpu().numpy(),
                            "full_pose": pose.detach().cpu().numpy(),
                        }
                    )
                    if coerced is not None:
                        self._smpl_seed = (coerced[0], coerced[1], "embedded_checkpoint")
                except Exception as exc:
                    self._smpl_error = f"SMPL neutral initialization failed: {exc}"
            else:
                try:
                    pose = _identity_pose6d()
                    rot = _rotation_6d_to_matrix_torch(
                        self._torch.from_numpy(pose).reshape(1, 24, 6)
                    )
                    betas = self._torch.zeros((1, 10), device=self.device)
                    output = self._smpl_model(
                        body_pose=rot[:, 1:],
                        global_orient=rot[:, :1],
                        betas=betas,
                        pose2rot=False,
                        return_full_pose=True,
                    )
                    source = {
                        "joints": getattr(output, "joints"),
                        "full_pose": getattr(output, "full_pose", rot),
                    }
                    self._smpl_seed = _coerce_smpl_initialization(source)
                except Exception as exc:
                    self._smpl_error = f"SMPL neutral initialization failed: {exc}"

    def _smpl_initialization(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """Return a SMPL-derived seed or an explicit deterministic fallback."""
        if self._smpl_seed is not None:
            return self._smpl_seed

        provider = _first_config_value(
            self.config,
            "smplInitializer",
            "smpl_initializer",
            "initializationProvider",
            "initialization_provider",
        )
        if provider is not None:
            try:
                if callable(provider):
                    value = provider()
                else:
                    method = getattr(provider, "get_initialization", None)
                    if method is None:
                        method = getattr(provider, "initialize", None)
                    value = method() if callable(method) else provider
                self._smpl_seed = _coerce_smpl_initialization(value)
                if self._smpl_seed is not None:
                    return self._smpl_seed
            except Exception as exc:
                self._smpl_error = f"SMPL initialization provider failed: {exc}"

        if self._allow_neutral_seed:
            # A neutral seed keeps the temporal core runnable when only the
            # checkpoint is present.  It is deliberately labelled as a
            # fallback; it is not presented as image-conditioned SMPL output.
            joints = np.zeros((17, 3), dtype=np.float32)
            self._smpl_seed = (joints, _identity_pose6d(), "neutral_fallback")
            return self._smpl_seed

        reason = self._smpl_error or (
            "no SMPL initialization was supplied and the WHAM checkpoint did not "
            "contain usable embedded body-model buffers"
        )
        raise RuntimeError(
            "WHAM SMPL-derived recurrent initialization is unavailable: " + reason +
            ". Supply smplInitialization/smplModelPath or enable a neutral seed explicitly."
        )

    def set_initialization_from_observation(self, observation: Any) -> bool:
        """Seed WHAM with a first-frame camera-space 3D observation.

        WHAM's released motion encoder expects ``init_kp3d`` in the same
        camera convention as its 2D input (COCO-17, image ``+Y`` down and
        camera ``+Z`` away).  A neutral SMPL seed is a valid tensor shape but
        has no relationship to the current subject or camera, so the
        recurrent rollout can converge to a collapsed/inverted body.  The
        built-in MediaPipe auxiliary already supplies a metric camera-space
        world observation; use it for the first-frame seed when no explicit
        SMPL initialization was configured.

        The method is deliberately a small adapter seam.  It accepts either
        ``FrameObservations`` or a mapping with ``landmarks3d`` and returns
        ``True`` only when at least a torso and one limb can be normalized.
        No image feature or network dependency is introduced here.
        """
        if self._smpl_seed_explicit:
            return False
        current_source = self._smpl_seed[2] if self._smpl_seed is not None else None
        if current_source not in (None, "embedded_checkpoint", "neutral_fallback"):
            return False

        points = None
        if isinstance(observation, Mapping):
            points = observation.get(
                "landmarks3d",
                observation.get(
                    "landmarks_3d",
                    observation.get("keypoints3d", observation.get("keypoints_3d")),
                ),
            )
        elif isinstance(observation, np.ndarray):
            # A direct ``(N, 3)`` landmark array is useful to callers that
            # already unpacked the MediaPipe result.  Do not mistake its
            # first row for the complete observation.
            points = observation if observation.ndim >= 2 else None
        elif isinstance(observation, (list, tuple)):
            # MediaPipePoseTracker's lightweight contract is a tuple of
            # ``(world_landmarks, normalized_landmarks, confidence)``.  The
            # native adapter is also used directly by the extractor, so
            # accept that contract here instead of requiring callers to
            # manufacture a FrameObservations object.  A plain list of point
            # rows remains valid as-is.
            if len(observation) == 3:
                try:
                    candidate = np.asarray(observation[0])
                except Exception:
                    candidate = np.asarray(())
                points = observation[0] if candidate.ndim >= 2 else observation
            else:
                points = observation
        else:
            points = getattr(observation, "landmarks_3d", None)
        if points is None:
            return False

        rows: List[List[float]] = []
        for point in list(points)[:33]:
            try:
                if isinstance(point, Mapping):
                    row = [point.get("x", np.nan), point.get("y", np.nan), point.get("z", np.nan)]
                elif isinstance(point, (list, tuple, np.ndarray)):
                    row = list(point[:3])
                else:
                    row = [point.x, point.y, point.z]
                rows.append([float(row[0]), float(row[1]), float(row[2])])
            except (AttributeError, IndexError, TypeError, ValueError):
                rows.append([np.nan, np.nan, np.nan])
        if len(rows) < 29:
            return False

        world = np.asarray(rows, dtype=np.float32)
        # Native WHAM's motion encoder uses COCO-17 ordering.  The first 17
        # rows of the embedded WHAM SMPL regressor are also COCO (nose, eyes,
        # ears, shoulders, elbows, wrists, hips, knees, ankles).
        coco_indices = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        seed = world[list(coco_indices), :3].copy()
        finite = np.isfinite(seed).all(axis=1)
        # Hips are the root anchor.  A missing face point is harmless, but a
        # missing pair of hips makes the camera-space origin ambiguous.
        if not (finite[11] and finite[12]):
            return False
        root = seed[[11, 12]].mean(axis=0)
        seed -= root[None, :]
        finite_count = int(np.count_nonzero(finite))
        if finite_count < 6:
            return False
        # Fill sparse detector gaps with the root rather than NaNs.  The mask
        # on the corresponding 2D input still tells WHAM those points are
        # missing; this keeps the recurrent initialization finite.
        seed[~finite] = 0.0
        self._smpl_seed = (seed.astype(np.float32), _identity_pose6d(), "mediapipe_observation")
        return True

    def get_mediapipe_auxiliary(self) -> Any:
        """Expose the detector when WHAM already owns a MediaPipe graph.

        The editor-side extractor can reuse this detector for the auxiliary
        3D fusion stream.  Creating another MediaPipe Tasks graph in the same
        Windows process is prone to XNNPACK/TFLite access violations, so this
        small ownership seam keeps one graph per extraction.
        """
        if str(self._detector_name or "").strip().lower() != "mediapipe":
            return None
        detector = self._detector
        return detector if callable(getattr(detector, "detect", None)) else None

    def _camera_for_sequence(self, frame_count: int) -> np.ndarray:
        source = self._camera_source
        motion_source_hint: Optional[str] = None
        if callable(source):
            source = source(frame_count)
        if isinstance(source, (str, os.PathLike)) and (
            str(source) == str(self._camera_asset_path or "")
            or str(source) == str(self._dpvo_asset_path or "")
        ):
            label = "camera" if str(source) == str(self._camera_asset_path or "") else "DPVO"
            try:
                suffix = Path(source).suffix.lower()
                payload = _load_local_asset_payload(source, label)
                motion_value = _camera_motion_value(payload)
                if motion_value is None:
                    if label == "camera" and _is_camera_calibration_payload(payload):
                        self._camera_calibration = payload
                        self._camera_motion_source = "calibration_only"
                    elif label == "DPVO":
                        self._dpvo_weights_available = suffix in (".pt", ".pth", ".tar", ".ckpt")
                        if self._dpvo_weights_available:
                            self._camera_motion_source = "dpvo_weights_only"
                            self._optional_errors.append(
                                "DPVO weights are installed but no camera poses were exported; "
                                "run the local DPVO runner or provide slamResults/cameraPoses."
                            )
                    source = None
                else:
                    source = motion_value
                    motion_source_hint = "dpvo" if label == "DPVO" else "camera_archive"
            except Exception as exc:
                self._optional_errors.append(f"{label} camera-motion asset could not be used: {exc}")
                source = None
                if self._strict_optional_assets:
                    raise
        # When only a camera/DPVO asset path was configured, inspect it lazily
        # at inference time.  Calibration files and raw DPVO checkpoints are
        # valid companion assets but do not contain per-frame angular motion.
        if source is None:
            candidate_paths = []
            if self._camera_asset_path and self._camera_source != self._camera_asset_path:
                candidate_paths.append(("camera", self._camera_asset_path))
            if self._dpvo_asset_path and self._camera_source != self._dpvo_asset_path:
                candidate_paths.append(("DPVO", self._dpvo_asset_path))
            for label, path in candidate_paths:
                try:
                    suffix = Path(path).suffix.lower()
                    payload = _load_local_asset_payload(path, label)
                    motion_value = _camera_motion_value(payload)
                    if motion_value is not None:
                        source = motion_value
                        motion_source_hint = "dpvo" if label == "DPVO" else "camera_archive"
                        break
                    if label == "camera" and _is_camera_calibration_payload(payload):
                        self._camera_calibration = payload
                        self._camera_motion_source = "calibration_only"
                    elif label == "DPVO":
                        # A .pth/.ckpt generally contains network weights; it
                        # becomes usable for world motion only after a DPVO
                        # runtime exports poses or a caller supplies a runner.
                        self._dpvo_weights_available = suffix in (".pt", ".pth", ".tar", ".ckpt")
                        if self._dpvo_weights_available:
                            self._camera_motion_source = "dpvo_weights_only"
                            self._optional_errors.append(
                                "DPVO weights are installed but no camera poses were exported; "
                                "run the local DPVO runner or provide slamResults/cameraPoses."
                            )
                except Exception as exc:
                    self._optional_errors.append(f"{label} camera-motion asset could not be used: {exc}")
                    if self._strict_optional_assets:
                        raise
        if isinstance(source, Mapping):
            source = _first_config_value(
                source,
                "angular_velocity",
                "cameraAngularVelocity",
                "camera_angular_velocity",
                "poses",
                "cameraPoses",
                "camera_poses",
                "slamResults",
                "slam_results",
            )
        if source is None:
            self._camera_angular_velocity = np.zeros((frame_count, 6), dtype=np.float32)
            if self._camera_motion_source not in ("calibration_only", "dpvo_weights_only"):
                self._camera_motion_source = "none"
            self._dpvo_verified = False
            self._camera_motion_provenance = {
                "source": self._camera_motion_source,
                "verified": False,
                "runner": self._camera_preprocess_runner_name,
                "asset": self._dpvo_asset_path if self._camera_motion_source == "dpvo_weights_only" else None,
            }
            return self._camera_angular_velocity
        try:
            self._camera_angular_velocity = _coerce_camera_angular_velocity(source, frame_count, self._fps)
        except Exception as exc:
            self._optional_errors.append(f"camera motion payload is invalid: {exc}")
            if self._strict_optional_assets:
                raise
            self._camera_angular_velocity = np.zeros((frame_count, 6), dtype=np.float32)
            self._camera_motion_source = "none"
            self._dpvo_verified = False
            self._camera_motion_provenance = {
                "source": "none",
                "verified": False,
                "runner": self._camera_preprocess_runner_name,
                "asset": None,
            }
            return self._camera_angular_velocity
        # Keep true DPVO output distinct from the bundled optical-flow
        # approximation and from an unlabelled caller-provided archive.
        if self._camera_preprocess_runner_name == "dpvo" or motion_source_hint == "dpvo":
            self._camera_motion_source = "dpvo"
            self._dpvo_verified = True
        elif self._camera_preprocess_runner_name == "local_optical_flow":
            self._camera_motion_source = "local_optical_flow"
            self._dpvo_verified = False
        elif self._camera_preprocess_runner_name == "configured_camera_pose":
            self._camera_motion_source = "configured_camera_pose"
            self._dpvo_verified = False
        else:
            self._camera_motion_source = "provided"
            self._dpvo_verified = False
        self._camera_motion_provenance = {
            "source": self._camera_motion_source,
            "verified": bool(self._dpvo_verified),
            "runner": self._camera_preprocess_runner_name,
            "asset": self._dpvo_asset_path if self._camera_motion_source == "dpvo" else (
                self._camera_source if isinstance(self._camera_source, (str, os.PathLike)) else None
            ),
        }
        return self._camera_angular_velocity

    def _image_features_for_sequence(
        self,
        frame_count: int,
        frames_bgr: Optional[Sequence[np.ndarray]] = None,
        timestamps_sec: Optional[Sequence[float]] = None,
        frame_indices: Optional[Sequence[int]] = None,
        frame_bboxes: Optional[Sequence[Any]] = None,
    ) -> Optional[np.ndarray]:
        source = self._feature_source
        # Once a runner has been rejected, never retry a configured local
        # descriptor/callable as if it were a learned WHAM feature source.
        # This guard is deliberately before source resolution because the
        # config may still contain the failed runner object/module path.
        if self._feature_fallback_active or str(self._feature_source_provenance or "").startswith(
            "local_frame_descriptor"
        ):
            if self._require_image_features:
                raise RuntimeError(
                    "WHAM image-feature integration was requested, but the configured "
                    "ViTPose/HMR2 runner was rejected; no learned image features were used"
                )
            return None
        if source is None:
            extractor = _first_config_value(
                self.config,
                "imageFeatureExtractor",
                "image_feature_extractor",
                "featureExtractor",
                "feature_extractor",
            )
            if extractor is None and frames_bgr is not None:
                # ``infer_sequence`` can be called with auto preprocessing
                # disabled.  Keep the long-standing archive/extractor path,
                # but still honor an inline runner when one was configured.
                extractor = self._resolve_inline_feature_runner()
            source = extractor
        if source is None:
            if self._feature_backbone_path:
                self._reject_local_feature_fallback(
                    "WHAM image-feature integrator was not given a verified "
                    "ViTPose/feature archive; the unverified local_frame_descriptor "
                    "fallback was rejected and was not fed into the learned "
                    "integrator; no learned image features were used.",
                    "image_feature_backbone_without_verified_extractor",
                )
                self._optional_errors.append(
                    "imageFeatureBackbonePath requires a verified offline frame-feature extractor; "
                    + self._feature_backbone_path
                )
            if self._require_image_features:
                raise RuntimeError(
                    "WHAM image-feature integration was requested, but no offline "
                    "image feature archive/extractor was configured"
                )
            return None
        source_runner = source
        if hasattr(source, "extract_sequence") or callable(source):
            try:
                source = self._invoke_image_feature_extractor(
                    source,
                    frames_bgr,
                    timestamps_sec if timestamps_sec is not None else [],
                    frame_indices if frame_indices is not None else list(range(frame_count)),
                    frame_bboxes,
                )
            except TypeError as invoke_error:
                # Preserve the historical callable(frame_count) archive seam
                # for explicit sources while keeping runner objects on the
                # frame-aware contract.
                if hasattr(source_runner, "extract_sequence"):
                    raise
                try:
                    source = source_runner(frame_count)
                except Exception:
                    feature_error = invoke_error
                    self._feature_runner_error = str(feature_error)
                    if source_runner is self._feature_runner:
                        self._feature_runner_active = False
                        self._feature_runner_kind = None
                    self._reject_local_feature_fallback(
                        "The configured ViTPose/HMR2 runner was incompatible with WHAM; "
                        "the unverified local_frame_descriptor fallback was rejected "
                        "and was not fed into the learned WHAM image-feature integrator; "
                        "the integrator will be skipped. "
                        f"Runner error: {feature_error}",
                        "image_feature_runner",
                    )
                    self._feature_runner_warning(
                        "ViTPose/HMR2 feature inference failed ("
                        + str(feature_error)
                        + "); unverified local_frame_descriptor was rejected."
                    )
                    if self._require_image_features:
                        raise RuntimeError(
                            "WHAM image-feature integration was requested, but the "
                            "configured ViTPose/HMR2 runner failed"
                        ) from feature_error
                    return None
            except Exception as feature_error:
                self._feature_runner_error = str(feature_error)
                if source_runner is self._feature_runner:
                    self._feature_runner_active = False
                    self._feature_runner_kind = None
                self._reject_local_feature_fallback(
                    "The configured ViTPose/HMR2 runner was incompatible with WHAM; "
                    "the unverified local_frame_descriptor fallback was rejected "
                    "and was not fed into the learned WHAM image-feature integrator; "
                    "the integrator will be skipped. "
                    f"Runner error: {feature_error}",
                    "image_feature_runner",
                )
                self._feature_runner_warning(
                    "ViTPose/HMR2 feature inference failed ("
                    + str(feature_error)
                    + "); unverified local_frame_descriptor was rejected."
                )
                if self._require_image_features:
                    raise RuntimeError(
                        "WHAM image-feature integration was requested, but the "
                        "configured ViTPose/HMR2 runner failed"
                    ) from feature_error
                return None
            self._record_feature_runner_metadata(
                source_runner,
                frame_indices if frame_indices is not None else list(range(frame_count)),
                frame_bboxes,
            )
        expected_dim = None
        integrator = getattr(self._model, "integrator", None)
        if integrator is not None:
            expected_dim = int(
                integrator.layer1.in_features
                - (self._architecture.get("d_embed", 512) + self._architecture.get("n_joints", 17) * 3)
            )
            if expected_dim <= 0:
                raise RuntimeError(
                    "WHAM Integrator exposes an invalid image-feature input dimension: "
                    f"{expected_dim}"
                )
        expected_fps = _first_config_value(
            self.config,
            "featureFps",
            "feature_fps",
            "targetFps",
            "target_fps",
            "fps",
            "frameRate",
        )
        if expected_fps is None and timestamps_sec is not None and len(timestamps_sec) > 1:
            try:
                requested_times = np.asarray(timestamps_sec, dtype=np.float64).reshape(-1)
                deltas = np.diff(requested_times)
                positive_deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
                if positive_deltas.size:
                    expected_fps = float(1.0 / np.median(positive_deltas))
            except (TypeError, ValueError, FloatingPointError):
                expected_fps = None
        expected_source_fps = _first_config_value(
            self.config,
            "sourceFps",
            "source_fps",
            "originalFps",
            "original_fps",
        )
        expected_trim_start = _first_config_value(
            self.config,
            "trimStart",
            "trim_start",
            "trimStartSec",
            "trim_start_sec",
        )
        expected_trim_end = _first_config_value(
            self.config,
            "trimEnd",
            "trim_end",
            "trimEndSec",
            "trim_end_sec",
        )
        expected_video_hash = _first_config_value(
            self.config,
            "videoHash",
            "video_hash",
            "sourceVideoHash",
            "source_video_hash",
        )
        expected_video_path = _first_config_value(
            self.config,
            "videoPath",
            "video_path",
            "sourceVideoPath",
            "source_video_path",
        )
        require_archive_metadata = _config_bool(
            self.config,
            "requireFeatureArchiveMetadata",
            "require_feature_archive_metadata",
            default=False,
        )
        try:
            values = _coerce_image_features(
                source,
                frame_count,
                expected_dim,
                frame_indices=frame_indices,
                expected_fps=expected_fps,
                expected_source_fps=expected_source_fps,
                expected_timestamps=timestamps_sec,
                expected_trim_start=expected_trim_start,
                expected_trim_end=expected_trim_end,
                expected_video_hash=expected_video_hash,
                expected_video_path=expected_video_path,
                require_archive_metadata=require_archive_metadata,
            )
        except _UnverifiedFeatureSourceError as feature_error:
            # A ``.pth``/``.pt`` extension is ambiguous: callers sometimes
            # point the feature setting at the downloaded ViTPose checkpoint.
            # Reject that payload explicitly and clear the source so a raw
            # model archive can never be passed to the learned integrator.
            self._feature_runner_error = str(feature_error)
            self._feature_source = None
            self._reject_local_feature_fallback(
                "The configured WHAM image-feature source is a raw model "
                "checkpoint, not a verified frame-feature archive; the "
                "unverified local_frame_descriptor fallback was rejected and "
                "was not fed into the learned WHAM image-feature integrator; "
                "the integrator will be skipped. "
                f"Source error: {feature_error}",
                "image_feature_checkpoint",
            )
            self._feature_runner_warning(
                "Raw ViTPose/HMR2 checkpoint was rejected as image features; "
                "unverified local_frame_descriptor was not consumed."
            )
            if self._require_image_features:
                raise RuntimeError(
                    "WHAM image-feature integration requires a verified frame-feature "
                    "archive; a raw model checkpoint was supplied"
                ) from feature_error
            return None
        except Exception as feature_error:
            if frame_count == 1:
                return None
            if hasattr(source_runner, "extract_sequence") or callable(source_runner):
                self._feature_runner_error = str(feature_error)
                if source_runner is self._feature_runner:
                    self._feature_runner_active = False
                    self._feature_runner_kind = None
                self._reject_local_feature_fallback(
                    "The configured ViTPose/HMR2 runner output was incompatible with WHAM; "
                    "the unverified local_frame_descriptor fallback was rejected "
                    "and was not fed into the learned WHAM image-feature integrator; "
                    "the integrator will be skipped. "
                    f"Runner error: {feature_error}",
                    "image_feature_runner",
                )
                self._feature_runner_warning(
                    "ViTPose/HMR2 feature output was incompatible ("
                    + str(feature_error)
                    + "); unverified local_frame_descriptor was rejected."
                )
                if self._require_image_features:
                    raise RuntimeError(
                        "WHAM image-feature integration was requested, but the "
                        "configured ViTPose/HMR2 runner output was incompatible"
                    ) from feature_error
                return None
            # A verified archive can still be stale, malformed, or aligned to
            # another trim.  Treat that as an explicit rejected image stage in
            # optional mode; never let a bad archive reach the Integrator and
            # never replace it with a local descriptor.
            self._feature_runner_error = str(feature_error)
            self._reject_local_feature_fallback(
                "The configured WHAM image-feature archive was rejected and was "
                "not fed into the learned image-feature integrator; "
                f"archive error: {feature_error}",
                "image_feature_archive",
            )
            if self._require_image_features:
                raise RuntimeError(
                    "WHAM image-feature integration requires a valid frame-aligned archive"
                ) from feature_error
            return None
        # Archive shape/finite checks above are the minimum verification gate
        # for a learned feature input.  Internally generated local descriptors
        # are never assigned this verified provenance.
        if self._feature_source_provenance == "none":
            self._feature_source_provenance = (
                "vitpose_inline" if self._feature_runner_active else "configured_extractor"
            )
        self._feature_source_verified = True
        return values

    def _inline_feature_runner_requested(self) -> bool:
        """Whether configuration asks for an inline ViTPose feature stage."""
        return any(
            _first_config_value(self.config, *names) is not None
            for names in (
                (
                    "imageFeatureRunner",
                    "image_feature_runner",
                    "vitposeRunner",
                    "vitpose_runner",
                    "featureRunner",
                    "feature_runner",
                ),
                (
                    "imageFeatureRunnerModule",
                    "image_feature_runner_module",
                    "vitposeRunnerModule",
                    "vitpose_runner_module",
                    "vitposeRunnerPath",
                    "vitpose_runner_path",
                ),
                (
                    "imageFeatureModelFactory",
                    "image_feature_model_factory",
                    "vitposeModelFactory",
                    "vitpose_model_factory",
                    "modelFactory",
                    "model_factory",
                ),
                (
                    "imageFeatureModelDefinition",
                    "image_feature_model_definition",
                    "vitposeModelDefinition",
                    "vitpose_model_definition",
                    "modelDefinition",
                    "model_definition",
                ),
                (
                    "imageFeatureConfigPath",
                    "image_feature_config_path",
                    "vitposeConfigPath",
                    "vitpose_config_path",
                    "modelConfigPath",
                    "model_config_path",
                ),
                (
                    "imageFeatureCheckpointPath",
                    "image_feature_checkpoint_path",
                    "vitposeCheckpointPath",
                    "vitpose_checkpoint_path",
                    "vitposeModelPath",
                    "vitpose_model_path",
                ),
                (
                    "imageFeatureRuntime",
                    "image_feature_runtime",
                    "vitposeRuntime",
                    "vitpose_runtime",
                ),
            )
        )

    def _feature_runner_warning(self, message: str) -> None:
        if message not in self._runtime_warnings:
            self._runtime_warnings.append(message)

    @staticmethod
    def _serializable_feature_metadata(value: Any) -> Any:
        """Convert runner metadata to values safe for JSON/export consumers."""
        if isinstance(value, Mapping):
            return {
                str(key): NativeWHAMAdapter._serializable_feature_metadata(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [NativeWHAMAdapter._serializable_feature_metadata(item) for item in value]
        if isinstance(value, np.ndarray):
            return NativeWHAMAdapter._serializable_feature_metadata(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _record_feature_runner_metadata(
        self,
        runner: Any,
        frame_indices: Sequence[int],
        frame_bboxes: Optional[Sequence[Any]] = None,
    ) -> None:
        """Capture runner provenance and the exact frames it adopted.

        A feature runner may expose ``get_metadata`` after inference.  When it
        does not, the adapter still records the input identity and assumes all
        rows were adopted; a crop-aware runner can override that decision by
        exposing ``personCrop`` and invalid boxes are then excluded.
        """
        try:
            raw_input_indices = list(frame_indices)
            input_indices = []
            for value in raw_input_indices:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError("boolean values are not frame indices")
                numeric = float(value)
                if not np.isfinite(numeric) or numeric != math.trunc(numeric):
                    raise ValueError(f"{value!r} is not an integer-like frame index")
                input_indices.append(int(numeric))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"frame_indices are not integer-like: {exc}") from exc
        if len(set(input_indices)) != len(input_indices):
            raise ValueError("frame_indices contain duplicate source frame IDs")
        metadata: Mapping[str, Any] = {}
        getter = getattr(runner, "get_metadata", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, Mapping):
                    metadata = value
            except Exception as exc:
                self._feature_runner_warning(
                    f"image feature runner metadata was unavailable: {exc}"
                )
        if not metadata:
            value = getattr(runner, "metadata", None)
            if isinstance(value, Mapping):
                metadata = value
        normalized = self._serializable_feature_metadata(metadata)
        if not isinstance(normalized, dict):
            normalized = {}
        self._feature_runner_metadata = normalized
        self._feature_input_frame_indices = input_indices

        adopted_value = normalized.get(
            "adoptedFrameIndices",
            normalized.get("adopted_frame_indices", normalized.get("featureFrameIndices")),
        )
        adopted: Optional[List[int]] = None
        if adopted_value is not None:
            try:
                adopted = []
                for value in adopted_value:
                    if isinstance(value, (bool, np.bool_)):
                        raise ValueError("boolean values are not frame indices")
                    numeric = float(value)
                    if not np.isfinite(numeric) or numeric != math.trunc(numeric):
                        raise ValueError(f"{value!r} is not an integer-like frame index")
                    adopted.append(int(numeric))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"image feature runner adoptedFrameIndices are not integer-like: {exc}"
                ) from exc
        if adopted is None:
            person_crop_value = normalized.get(
                "personCrop",
                normalized.get("person_crop", getattr(runner, "person_crop", False)),
            )
            person_crop = (
                str(person_crop_value).strip().lower()
                not in ("", "0", "false", "no", "off", "none", "null")
                if isinstance(person_crop_value, str)
                else bool(person_crop_value)
            )
            if person_crop and frame_bboxes is not None:
                adopted = [
                    input_indices[position]
                    for position, bbox in enumerate(frame_bboxes)
                    if position < len(input_indices) and bbox is not None
                ]
            else:
                adopted = list(input_indices)
        valid_ids = set(input_indices)
        if len(set(adopted)) != len(adopted) or any(index not in valid_ids for index in adopted):
            raise ValueError(
                "image feature runner adoptedFrameIndices contain duplicate or unknown frame IDs"
            )
        # Rows may be sparse when person crops are unavailable, but they must
        # retain source order.  A reordered metadata list would make a valid
        # (N,D) tensor look finite while pairing features with the wrong frame.
        ordered_adopted = [index for index in input_indices if index in set(adopted)]
        if list(adopted) != ordered_adopted:
            raise ValueError(
                "image feature runner adoptedFrameIndices are not aligned with input frame order"
            )
        self._feature_adopted_frame_indices = [
            index for index in adopted if index in valid_ids
        ]
        self._feature_runner_metadata.setdefault(
            "inputFrameIndices", list(input_indices)
        )
        self._feature_runner_metadata.setdefault(
            "adoptedFrameIndices", list(self._feature_adopted_frame_indices)
        )
        self._feature_runner_metadata.setdefault(
            "adoptedFrameCount", len(self._feature_adopted_frame_indices)
        )

    def _reject_local_feature_fallback(self, reason: str, asset: Optional[str] = None) -> None:
        """Record the explicit local-descriptor decision without producing features."""
        self._feature_fallback_active = True
        self._feature_fallback_kind = "local_frame_descriptor"
        self._feature_fallback_consumed = False
        self._feature_fallback_reason = str(reason)
        self._feature_source_provenance = "local_frame_descriptor_rejected"
        self._feature_source_verified = False
        reason_text = str(reason).lower()
        if "dimension" in reason_text or "width" in reason_text:
            self._feature_error_code = "feature_dim_mismatch"
            self._feature_next_action = "Select a runner whose image encoder output matches the WHAM Integrator dimension."
        elif "frame" in reason_text and ("align" in reason_text or "count" in reason_text):
            self._feature_error_code = "feature_frame_count_mismatch"
            self._feature_next_action = "Keep runner output rows aligned with the requested source frame indices."
        elif "non-finite" in reason_text or "nan" in reason_text or "inf" in reason_text:
            self._feature_error_code = "feature_non_finite"
            self._feature_next_action = "Fix preprocessing/model output so every feature value is finite."
        elif "checkpoint" in reason_text and "raw" in reason_text:
            self._feature_error_code = "checkpoint_not_executable"
            self._feature_next_action = "Configure an executable model factory or provide a verified feature archive."
        elif "runner" in reason_text or "extractor" in reason_text:
            self._feature_error_code = "official_runner_inference_failed"
            self._feature_next_action = "Run image-feature Preflight and configure a compatible HMR2/ViTPose runtime."
        else:
            self._feature_error_code = "feature_contract_failed"
            self._feature_next_action = "Run image-feature Preflight and inspect the recorded runner error."
        if asset:
            self._incompatible_assets.append(str(asset))

    @staticmethod
    def _invoke_image_feature_extractor(
        extractor: Any,
        frames: Sequence[np.ndarray],
        timestamps: Sequence[float],
        frame_indices: Sequence[int],
        frame_bboxes: Optional[Sequence[Any]] = None,
    ) -> Any:
        """Invoke an extractor while preserving the source frame IDs."""
        method = getattr(extractor, "extract_sequence", None)
        if not callable(method):
            method = extractor
        if not callable(method):
            raise TypeError(
                "imageFeatureExtractor/imageFeatureRunner must be callable or expose "
                "extract_sequence"
            )
        if frame_bboxes is not None:
            try:
                return method(
                    frames,
                    list(timestamps),
                    list(frame_indices),
                    frame_bboxes=list(frame_bboxes),
                )
            except TypeError:
                # A few lightweight adapters expose the bbox argument as a
                # fourth positional parameter rather than a keyword.  Try
                # that shape before preserving the historical three-argument
                # extractor contract.
                try:
                    return method(
                        frames,
                        list(timestamps),
                        list(frame_indices),
                        list(frame_bboxes),
                    )
                except TypeError:
                    pass
        try:
            return method(frames, list(timestamps), list(frame_indices))
        except TypeError:
            return method(frames)

    def _resolve_inline_feature_runner(self) -> Any:
        """Resolve and initialize the optional in-process ViTPose runner.

        The runner module is deliberately imported only after a WHAM sequence
        needs image features.  This keeps the default runtime torch-free and
        lets callers inject either a ready ``extract_sequence`` object or a
        model factory/checkpoint pair.  Any import, construction, or checkpoint
        compatibility failure returns ``None`` so the caller can retain the
        explicit image-feature fallback.  A raw/local descriptor is never
        passed to WHAM's learned image-feature integrator.
        """
        if self._feature_runner_resolution_attempted:
            return self._feature_runner
        self._feature_runner_resolution_attempted = True

        runner_value = _first_config_value(
            self.config,
            "imageFeatureRunner",
            "image_feature_runner",
            "vitposeRunner",
            "vitpose_runner",
            "featureRunner",
            "feature_runner",
        )
        runner_module_value = _first_config_value(
            self.config,
            "imageFeatureRunnerModule",
            "image_feature_runner_module",
            "vitposeRunnerModule",
            "vitpose_runner_module",
            "vitposeRunnerPath",
            "vitpose_runner_path",
        )
        model_factory_value = _first_config_value(
            self.config,
            "imageFeatureModelFactory",
            "image_feature_model_factory",
            "vitposeModelFactory",
            "vitpose_model_factory",
            "modelFactory",
            "model_factory",
        )
        model_definition = _first_config_value(
            self.config,
            "imageFeatureModelDefinition",
            "image_feature_model_definition",
            "vitposeModelDefinition",
            "vitpose_model_definition",
            "modelDefinition",
            "model_definition",
        )
        model_config_path = _first_config_value(
            self.config,
            "imageFeatureConfigPath",
            "image_feature_config_path",
            "vitposeConfigPath",
            "vitpose_config_path",
            "modelConfigPath",
            "model_config_path",
        )
        runner_config = _first_config_value(
            self.config,
            "imageFeatureRunnerConfig",
            "image_feature_runner_config",
            "vitposeRunnerConfig",
            "vitpose_runner_config",
        )
        factory_loads_checkpoint = _first_config_value(
            self.config,
            "imageFeatureFactoryLoadsCheckpoint",
            "image_feature_factory_loads_checkpoint",
            "vitposeFactoryLoadsCheckpoint",
            "vitpose_factory_loads_checkpoint",
            "factoryLoadsCheckpoint",
            "factory_loads_checkpoint",
        )
        if isinstance(runner_config, Mapping):
            runner_config = dict(runner_config)
        if factory_loads_checkpoint is not None:
            if runner_config is None:
                runner_config = {}
            if isinstance(runner_config, Mapping):
                runner_config.setdefault(
                    "factory_loads_checkpoint", factory_loads_checkpoint
                )
        declared_runtime = _first_config_value(
            self.config,
            "imageFeatureRuntime",
            "image_feature_runtime",
            "vitposeRuntime",
            "vitpose_runtime",
            "runtimeKind",
            "runtime_kind",
        )
        # ``imageFeatureRunner`` may be a compact mapping such as
        # ``{"runtime": "hmr2"}``; resolve that declaration before choosing
        # the factory so it cannot silently take the generic ViTPose path.
        if declared_runtime is None and isinstance(runner_config, Mapping):
            declared_runtime = _first_config_value(
                runner_config,
                "runtime",
                "runtimeKind",
                "runtime_kind",
                "runnerKind",
                "runner_kind",
            )
        hmr2_runtime_path = _first_config_value(
            self.config,
            "HMR2RuntimePath",
            "hmr2RuntimePath",
            "runtimePath",
            "runtime_path",
        )
        hmr2_body_model_path = _first_config_value(
            self.config,
            "HMR2BodyModelPath",
            "hmr2BodyModelPath",
            "bodyModelPath",
            "body_model_path",
        )
        runner_kind = _first_config_value(
            self.config,
            "WHAMRunnerKind",
            "whamRunnerKind",
            "runnerKind",
            "runner_kind",
        )
        if declared_runtime is None and (
            hmr2_runtime_path is not None
            or str(runner_kind or "").strip().lower() in {"official_hmr2", "hmr2", "hmr2_image_features"}
        ):
            declared_runtime = "hmr2"
        if declared_runtime is not None:
            if runner_config is None:
                runner_config = {}
            if isinstance(runner_config, Mapping):
                runner_config.setdefault("runtime", declared_runtime)
        if hmr2_runtime_path is not None:
            if runner_config is None:
                runner_config = {}
            if isinstance(runner_config, Mapping):
                runner_config.setdefault("runtime_path", hmr2_runtime_path)
                runner_config.setdefault("runtimePath", hmr2_runtime_path)
        if hmr2_body_model_path is not None:
            if runner_config is None:
                runner_config = {}
            if isinstance(runner_config, Mapping):
                runner_config.setdefault("body_model_path", hmr2_body_model_path)
                runner_config.setdefault("bodyModelPath", hmr2_body_model_path)
        checkpoint_path = _first_config_value(
            self.config,
            "imageFeatureCheckpointPath",
            "image_feature_checkpoint_path",
            "vitposeCheckpointPath",
            "vitpose_checkpoint_path",
            "vitposeModelPath",
            "vitpose_model_path",
        ) or self._feature_backbone_path

        # A downloaded ViTPose backbone is the opt-in signal for the bundled
        # inline module.  Do not import it for ordinary native WHAM fixtures
        # that intentionally run without image-side features.
        explicit_request = self._inline_feature_runner_requested()
        if not explicit_request and not self._feature_backbone_path:
            return None

        self._feature_runner_checkpoint = (
            str(checkpoint_path) if checkpoint_path is not None else None
        )
        self._feature_runner_model_definition = model_definition
        self._feature_runner_config_path = model_config_path
        module_label = str(runner_module_value or "pose_pipeline.vitpose_runner")
        self._feature_runner_module = module_label

        try:
            candidate = None
            factory = None
            module = None
            if runner_value is not None:
                candidate = _load_configured_symbol(runner_value, "ViTPose runner")
                if isinstance(candidate, Mapping):
                    # Accept ``imageFeatureRunner={runtime: hmr2, ...}`` as
                    # a compact config while keeping the default runner
                    # module as the implementation hook.
                    if runner_config is None:
                        runner_config = dict(candidate)
                    if declared_runtime is None:
                        declared_runtime = _first_config_value(
                            candidate,
                            "runtime",
                            "runtimeKind",
                            "runtime_kind",
                            "runnerKind",
                            "runner_kind",
                        )
                elif isinstance(candidate, type):
                    factory = candidate
                elif callable(getattr(candidate, "extract_sequence", None)):
                    self._feature_runner = candidate
                elif callable(candidate):
                    # A function/class supplied as ``vitposeRunner`` is a
                    # constructor hook.  A ready runner instance is handled
                    # above by its ``extract_sequence`` method.
                    factory = candidate
                else:
                    module = candidate
            if self._feature_runner is None:
                if module is None:
                    loaded = _load_configured_symbol(module_label, "ViTPose runner")
                    if isinstance(loaded, type) or (
                        callable(loaded)
                        and not callable(getattr(loaded, "extract_sequence", None))
                    ):
                        factory = loaded
                    else:
                        module = loaded
                if str(declared_runtime or "").strip().lower() == "hmr2":
                    factory = factory or getattr(module, "create_hmr2_runner", None)
                factory = factory or getattr(module, "create_vitpose_runner", None)
                factory = factory or getattr(module, "create_runner", None)
                factory = factory or getattr(module, "ViTPoseFeatureRunner", None)
                factory = factory or getattr(module, "Runner", None)
                if factory is None:
                    raise TypeError(
                        f"{module_label} exposes no create_vitpose_runner, create_runner, "
                        "or ViTPoseFeatureRunner factory"
                    )

                if isinstance(model_factory_value, (str, os.PathLike)):
                    model_factory_value = _load_configured_symbol(
                        model_factory_value, "ViTPose model factory"
                    )
                factory_kwargs = {
                    "config": runner_config,
                    "checkpoint_path": checkpoint_path,
                    "model_definition": model_definition,
                    "config_path": model_config_path,
                    "model_factory": model_factory_value,
                    "torch_module": self._torch,
                    "device": self.device,
                }
                try:
                    self._feature_runner = factory(**factory_kwargs)
                except TypeError as keyword_error:
                    # Small project factories often accept one config mapping
                    # instead of keyword arguments.  Preserve support for that
                    # form without imposing a dependency on a runner package.
                    try:
                        self._feature_runner = factory(factory_kwargs)
                    except TypeError:
                        raise keyword_error

            initialize = getattr(self._feature_runner, "initialize", None)
            if callable(initialize) and initialize() is False:
                raise RuntimeError("ViTPose runner initialization returned false")
            if not _runner_like(self._feature_runner):
                raise TypeError(
                    "ViTPose runner must expose extract_sequence(frames, timestamps, indices) "
                    "or be callable"
                )
            self._feature_runner_active = True
            self._feature_runner_kind = "vitpose_inline"
            return self._feature_runner
        except Exception as exc:
            self._feature_runner = None
            self._feature_runner_active = False
            self._feature_runner_error = str(exc)
            self._reject_local_feature_fallback(
                "No verified ViTPose/WHAM image-feature runner is available; "
                "the unverified local_frame_descriptor fallback was rejected and "
                "was not fed into the learned WHAM image-feature integrator; "
                "the integrator will be skipped. "
                f"Runner error: {exc}",
                "image_feature_runner",
            )
            self._feature_runner_metadata = {
                "runtime": "hmr2" if any(
                    str(_first_config_value(self.config, key) or "").strip().lower()
                    == "hmr2"
                    for key in ("imageFeatureRuntime", "image_feature_runtime")
                ) else "vitpose",
                "runnerKind": "unavailable",
                "module": module_label,
                "checkpointPath": self._feature_runner_checkpoint,
                "modelDefinition": self._feature_runner_model_definition,
                "error": str(exc),
                "inputFrameIndices": list(self._feature_input_frame_indices),
                "adoptedFrameIndices": [],
                "adoptedFrameCount": 0,
            }
            self._feature_runner_warning(
                "ViTPose inline runner unavailable ("
                + module_label
                + ": "
                + str(exc)
                + "); unverified local_frame_descriptor was rejected; configure a compatible "
                "imageFeatureExtractor, or provide a local ViTPose checkpoint, modelDefinition, "
                "and modelFactory/model runner."
            )
            return None

    def _prepare_local_runtime(
        self,
        frames_bgr: Sequence[np.ndarray],
        timestamps_sec: Sequence[float],
        frame_indices: Sequence[int],
        frame_bboxes: Optional[Sequence[Any]] = None,
        person_bboxes: Optional[Sequence[Any]] = None,
    ) -> None:
        """Generate per-video WHAM inputs when only downloaded weights exist.

        WHAM's official preprocessing is intentionally optional because its
        research repositories are not a dependency of the offline TexMotion
        runtime.  A configured/verified extractor or runner may produce an
        aligned feature archive, while a raw ViTPose checkpoint without one
        is recorded as an explicit fallback.  TexMotion never synthesizes a
        local descriptor and feeds it into WHAM's learned image integrator.
        """
        if frame_bboxes is not None and person_bboxes is not None:
            raise ValueError("Provide only one of frame_bboxes or person_bboxes")
        resolved_bboxes = frame_bboxes if frame_bboxes is not None else person_bboxes
        if resolved_bboxes is not None:
            resolved_bboxes = list(resolved_bboxes)
            if len(resolved_bboxes) != len(frames_bgr):
                raise ValueError(
                    f"frame_bboxes must have length {len(frames_bgr)}"
                )
        if len(frame_indices) != len(frames_bgr):
            raise ValueError(
                f"frame_indices must have length {len(frames_bgr)}"
            )
        try:
            normalized_indices = [int(value) for value in frame_indices]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"frame_indices are not integer-like: {exc}") from exc
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("frame_indices contain duplicate source frame IDs")
        if normalized_indices != list(frame_indices):
            raise ValueError("frame_indices must contain integer source frame IDs")
        frame_indices = normalized_indices
        if len(timestamps_sec) != len(frames_bgr):
            raise ValueError(
                f"timestamps_sec must have length {len(frames_bgr)}"
            )
        sequence_signature = _sequence_cache_identity(
            frames_bgr, timestamps_sec, frame_indices
        )
        self._feature_input_frame_indices = [int(index) for index in frame_indices]
        if not frames_bgr:
            self._feature_adopted_frame_indices = []
            return
        if not self.config.get("autoPreprocess", self.config.get("auto_preprocess", True)):
            return
        configured_feature_extractor = _first_config_value(
            self.config,
            "imageFeatureExtractor",
            "image_feature_extractor",
            "featureExtractor",
            "feature_extractor",
        )
        needs_features = bool(
            self._feature_source is None
            and self._loaded_components.get("imageFeatureIntegrator", False)
            and (
                self._feature_backbone_path
                or configured_feature_extractor is not None
                or self._inline_feature_runner_requested()
            )
        )
        # The backend performs a one-frame smoke inference before processing
        # the requested clip.  That smoke pass may create an auto-preprocess
        # camera cache containing exactly one pose.  Treat that generated
        # cache as stale when the next call contains a different number of
        # frames; otherwise the trajectory decoder receives ``(1, 6)`` for a
        # multi-frame sequence and fails at frame index 1.
        camera_motion_rows_match = bool(
            self._camera_angular_velocity is not None
            and np.asarray(self._camera_angular_velocity).ndim >= 2
            and int(np.asarray(self._camera_angular_velocity).shape[0]) == len(frames_bgr)
            and self._camera_preprocess_signature == sequence_signature
        )
        generated_camera_needs_refresh = bool(
            self._auto_camera_preprocess_used and not camera_motion_rows_match
        )
        needs_camera = bool(
            (
                self._camera_source is None
                or str(self._camera_source) == str(self._camera_asset_path or "")
                or generated_camera_needs_refresh
            )
            and self._dpvo_asset_path
            and self._enable_trajectory
            and self._loaded_components.get("trajectoryDecoder", False)
        )
        if not needs_features and not needs_camera:
            return
        try:
            from pose_pipeline.wham_preprocess import (
                estimate_local_camera_poses,
                run_wham_preprocess,
            )

            output_dir = self._preprocess_directory
            if not output_dir:
                root = self.model_directory or str(Path(self.model_path).parent)
                output_dir = str(Path(root) / "wham_preprocess")
            root = Path(output_dir).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            times = np.asarray(timestamps_sec, dtype=np.float32)
            indices = list(frame_indices)
            if len(indices) != len(frames_bgr):
                raise ValueError("WHAM preprocessing frame indices are not aligned")

            feature_extractor = None
            feature_runner_name = None
            if needs_features:
                if configured_feature_extractor is not None:
                    feature_extractor = configured_feature_extractor
                    feature_runner_name = "configured"
                else:
                    feature_extractor = self._resolve_inline_feature_runner()
                    if feature_extractor is not None:
                        feature_runner_name = self._feature_runner_kind or "vitpose_inline"
                    else:
                        # A compact descriptor has no learned WHAM training
                        # provenance.  Keep extraction alive by omitting the
                        # optional integrator input and expose this decision
                        # through metadata instead of silently fabricating a
                        # learned feature stream.
                        feature_extractor = None
                        feature_runner_name = "unverified_local_descriptor_rejected"
                        if not self._feature_fallback_reason:
                            fallback_reason = (
                                "No verified ViTPose/WHAM image-feature runner is configured; "
                                "the unverified local_frame_descriptor fallback was rejected "
                                "and was not fed into the learned WHAM image-feature integrator; "
                                "the integrator will be skipped."
                            )
                            self._reject_local_feature_fallback(
                                fallback_reason, "image_feature_runner"
                            )
                        else:
                            self._feature_fallback_active = True
                            self._feature_fallback_kind = "local_frame_descriptor"
                            self._feature_fallback_consumed = False
                            self._incompatible_assets.append("image_feature_runner")

                if feature_extractor is not None:
                    configured_source = feature_extractor

                    def feature_extractor(
                        images,
                        extractor_times,
                        extractor_indices,
                        frame_bboxes=None,
                    ):
                        result = self._invoke_image_feature_extractor(
                            configured_source,
                            images,
                            extractor_times,
                            extractor_indices,
                            frame_bboxes,
                        )
                        self._record_feature_runner_metadata(
                            configured_source,
                            extractor_indices,
                            frame_bboxes,
                        )
                        return result

            camera_extractor = None
            configured_camera_extractor = _first_config_value(
                self.config,
                "cameraPoseExtractor",
                "camera_pose_extractor",
                "dpvoRunner",
                "dpvo_runner",
            )
            if needs_camera:
                if configured_camera_extractor is not None:
                    camera_extractor = configured_camera_extractor
                else:
                    calibration = self._camera_calibration
                    if calibration is None and self._camera_asset_path:
                        try:
                            payload = _load_local_asset_payload(
                                self._camera_asset_path, "camera calibration"
                            )
                            if _is_camera_calibration_payload(payload):
                                calibration = payload
                                self._camera_calibration = payload
                        except Exception as exc:
                            self._runtime_warnings.append(
                                f"camera calibration could not be loaded for local runner: {exc}"
                            )

                    def camera_extractor(images, extractor_times, extractor_indices):
                        del extractor_indices
                        return estimate_local_camera_poses(
                            images, extractor_times, calibration
                        )

            def run_preprocess(feature_stage: Any, runner_name: Optional[str]) -> dict:
                metadata = {
                    "backend": "texmotion_native_wham",
                    "frameIndices": [int(index) for index in indices],
                    "featureInputFrameIndices": [int(index) for index in indices]
                    if needs_features else [],
                    "featureBboxCount": (
                        int(sum(bbox is not None for bbox in resolved_bboxes))
                        if needs_features and resolved_bboxes is not None else 0
                    ),
                    "imageFeatureBackbonePath": self._feature_backbone_path,
                    "dpvoModelPath": self._dpvo_asset_path,
                    "featureRunner": (
                        self._feature_fallback_kind
                        if runner_name == "unverified_local_descriptor_rejected"
                        else runner_name
                    ) if needs_features else None,
                    "featureRunnerStatus": (
                        "rejected"
                        if runner_name == "unverified_local_descriptor_rejected"
                        else ("active" if runner_name else "not_configured")
                    ) if needs_features else None,
                    "featureRunnerConsumed": bool(
                        runner_name not in (None, "unverified_local_descriptor_rejected")
                    ) if needs_features else False,
                    "imageFeatureRunner": runner_name if needs_features else None,
                    "imageFeatureRunnerModule": self._feature_runner_module,
                    "imageFeatureRunnerCheckpoint": self._feature_runner_checkpoint,
                    "cameraRunner": (
                        "dpvo" if _first_config_value(
                            self.config, "dpvoRunner", "dpvo_runner"
                        ) is not None else (
                            "configured_camera_pose" if configured_camera_extractor is not None
                            else "local_optical_flow"
                        )
                    ) if needs_camera else None,
                }
                if feature_stage is None and camera_extractor is None:
                    # Keep an aligned diagnostic manifest even when an
                    # optional image stage was rejected.  Calling the normal
                    # writer with no stages would raise before it could record
                    # the explicit fallback decision.
                    manifest = root / "manifest.json"
                    manifest.write_text(
                        json.dumps({
                            **metadata,
                            "frames": len(frames_bgr),
                            "timestamps": times.tolist(),
                        }, indent=2),
                        encoding="utf-8",
                    )
                    return {
                        "frames": len(frames_bgr),
                        "timestamps": times.tolist(),
                        "manifestPath": str(manifest),
                    }
                return run_wham_preprocess(
                    frames_bgr,
                    root,
                    image_feature_extractor=feature_stage,
                    camera_pose_extractor=camera_extractor,
                    timestamps=times,
                    frame_indices=indices,
                    frame_bboxes=resolved_bboxes if feature_stage is not None else None,
                    metadata=metadata,
                )

            try:
                result = run_preprocess(feature_extractor, feature_runner_name)
                if feature_runner_name == "vitpose_inline":
                    feature_shape = result.get("featureShape") if isinstance(result, dict) else None
                    expected_width = int(self._architecture.get("d_feat", 0) or 0)
                    actual_width = (
                        int(feature_shape[1])
                        if isinstance(feature_shape, (list, tuple)) and len(feature_shape) == 2
                        else 0
                    )
                    if expected_width > 0 and actual_width != expected_width:
                        raise ValueError(
                            "inline ViTPose feature width is incompatible with WHAM: "
                            f"expected {expected_width}, got {actual_width}"
                        )
            except Exception as feature_error:
                # A runner can construct successfully and still reject the
                # first batch (for example, a checkpoint with the wrong
                # feature width).  Keep extraction alive, but do not replace
                # the incompatible learned input with an unverified local
                # descriptor.
                if feature_runner_name not in ("vitpose_inline", "configured"):
                    raise
                self._feature_runner_error = str(feature_error)
                if feature_runner_name == "vitpose_inline":
                    self._feature_runner_active = False
                    self._feature_runner_kind = None
                fallback_label = (
                    "configured ViTPose/HMR2 runner"
                    if feature_runner_name == "configured"
                    else "configured ViTPose runner"
                )
                self._reject_local_feature_fallback(
                    f"The {fallback_label} was incompatible with WHAM; "
                    "the unverified local_frame_descriptor fallback was rejected "
                    "and was not fed into the learned WHAM image-feature integrator; "
                    "the integrator will be skipped. "
                    f"Runner error: {feature_error}",
                    "image_feature_runner",
                )
                self._feature_runner_warning(
                    f"{fallback_label.capitalize()} failed during feature extraction ("
                    + str(feature_error)
                    + "); unverified local_frame_descriptor was rejected. Check the local "
                    "checkpoint/model definition and expected WHAM feature dimension."
                )
                feature_runner_name = "unverified_local_descriptor_rejected"
                result = run_preprocess(None, feature_runner_name)
            self._preprocess_manifest_path = result.get("manifestPath")
            if needs_features and result.get("imageFeaturePath"):
                self._feature_source = result["imageFeaturePath"]
                if feature_runner_name == "unverified_local_descriptor_rejected":
                    # A rejected stage must never become a source merely
                    # because an older preprocess cache happens to exist.
                    self._feature_source = None
                else:
                    self._feature_source_provenance = (
                        "configured_extractor" if feature_runner_name == "configured" else "vitpose_inline"
                    )
                    self._feature_source_verified = True
                    self._feature_fallback_active = False
                    self._feature_fallback_kind = None
                    self._feature_fallback_consumed = False
                    self._auto_preprocess_used = True
                    self._auto_feature_preprocess_used = True
                    self._feature_preprocess_runner_name = feature_runner_name
            if needs_features and feature_runner_name != "unverified_local_descriptor_rejected":
                if not self._feature_input_frame_indices:
                    self._feature_input_frame_indices = list(indices)
                if "adoptedFrameIndices" not in self._feature_runner_metadata:
                    self._feature_adopted_frame_indices = list(indices)
                if isinstance(result, dict):
                    feature_shape = result.get("featureShape")
                    if isinstance(feature_shape, (list, tuple)):
                        self._feature_runner_metadata.setdefault(
                            "featureShape", [int(value) for value in feature_shape]
                        )
                        if len(feature_shape) > 1:
                            self._feature_runner_metadata.setdefault(
                                "featureDim", int(feature_shape[1])
                            )
                self._feature_runner_metadata.setdefault(
                    "inputFrameIndices", list(self._feature_input_frame_indices)
                )
                self._feature_runner_metadata.setdefault(
                    "adoptedFrameIndices", list(self._feature_adopted_frame_indices)
                )
                self._feature_runner_metadata.setdefault(
                    "adoptedFrameCount", len(self._feature_adopted_frame_indices)
                )
            manifest_path = result.get("manifestPath") if isinstance(result, dict) else None
            if needs_features and manifest_path:
                try:
                    manifest_file = Path(manifest_path)
                    manifest_payload = json.loads(manifest_file.read_text(encoding="utf-8"))
                    if not isinstance(manifest_payload, dict):
                        manifest_payload = {}
                    manifest_payload.update({
                        "featureRunnerMetadata": dict(self._feature_runner_metadata),
                        "featureInputFrameIndices": list(self._feature_input_frame_indices),
                        "featureAdoptedFrameIndices": list(self._feature_adopted_frame_indices),
                        "featureAdoptedFrameCount": len(self._feature_adopted_frame_indices),
                    })
                    manifest_file.write_text(
                        json.dumps(manifest_payload, indent=2), encoding="utf-8"
                    )
                except Exception as manifest_error:
                    self._feature_runner_warning(
                        f"image feature runner metadata could not be persisted: {manifest_error}"
                    )
            if needs_camera and result.get("cameraPosePath"):
                self._camera_source = result["cameraPosePath"]
                self._camera_preprocess_signature = sequence_signature
                self._auto_preprocess_used = True
                self._auto_camera_preprocess_used = True
                self._dpvo_weights_available = bool(self._dpvo_asset_path)
                configured_dpvo_runner = _first_config_value(
                    self.config, "dpvoRunner", "dpvo_runner"
                ) is not None
                self._camera_preprocess_runner_name = (
                    "dpvo" if configured_dpvo_runner else (
                        "configured_camera_pose" if configured_camera_extractor is not None
                        else "local_optical_flow"
                    )
                )
                if self._camera_preprocess_runner_name == "dpvo":
                    self._camera_motion_source = "dpvo"
                    self._dpvo_verified = True
                elif self._camera_preprocess_runner_name == "local_optical_flow":
                    self._camera_motion_source = "local_optical_flow"
                    self._dpvo_verified = False
                if configured_camera_extractor is None:
                    self._runtime_warnings.append(
                        "DPVO checkpoint detected; camera poses were generated by the bundled "
                        "offline optical-flow runner because no external DPVO runtime was configured."
                    )
            if needs_features and configured_feature_extractor is None and not self._feature_runner_active:
                self._runtime_warnings.append(
                    "ViTPose checkpoint detected; no verified inline feature extractor was configured. "
                    "The unverified local descriptor was rejected and learned image features were skipped."
                )
        except Exception as exc:
            message = f"WHAM per-video preprocessing failed: {exc}"
            self._optional_errors.append(message)
            if self._strict_optional_assets:
                raise RuntimeError(message) from exc

    def _root_initialization(self, pose6d: np.ndarray) -> np.ndarray:
        configured = _first_config_value(
            self.config,
            "initialRoot",
            "initRoot",
            "initial_root",
            "init_root",
        )
        if configured is not None:
            values = np.asarray(
                configured.detach().cpu().numpy() if hasattr(configured, "detach") else configured,
                dtype=np.float32,
            )
            if values.shape[-2:] == (3, 3):
                values = _matrix_to_rotation_6d_numpy(values)
            elif values.shape[-1] != 6:
                raise ValueError("WHAM initial root must be a 6D rotation or 3x3 matrix")
            return values.reshape(1, 1, 6)
        return pose6d[:1, :6].reshape(1, 1, 6)

    def _decode_smpl(self, pose: "torch.Tensor", betas: "torch.Tensor") -> Optional[Dict[str, Any]]:
        if self._smpl_model is None:
            return None
        if isinstance(self._smpl_model, EmbeddedSMPL):
            return self._smpl_model(pose, betas)
        model = self._smpl_model
        try:
            # External smplx-style models use axis-angle/matrix arguments and
            # return a ModelOutput.  Keep this adapter optional and avoid
            # importing smplx unless a caller explicitly supplied it.
            rot = _rotation_6d_to_matrix_torch(pose.reshape(-1, 24, 6))
            output = model(
                body_pose=rot[:, 1:],
                global_orient=rot[:, :1],
                betas=betas.reshape(-1, 10),
                pose2rot=False,
                return_full_pose=True,
            )
            result = {
                name: getattr(output, name)
                for name in ("vertices", "joints", "full_pose", "global_orient", "body_pose")
                if getattr(output, name, None) is not None
            }
            if "joints" in result:
                result["joints"] = result["joints"].reshape(pose.shape[0], pose.shape[1], -1, 3)
            return result
        except Exception as exc:
            self._smpl_error = f"configured SMPL decoder failed: {exc}"
            if self._strict_optional_assets:
                raise RuntimeError(self._smpl_error) from exc
            return None

    def initialize(self) -> bool:
        self._error = None
        self._preflight_status = "loading"
        self._preflight_error = None
        self._optional_errors = []
        self._runtime_warnings = []
        self._preprocess_manifest_path = None
        self._auto_preprocess_used = False
        self._auto_feature_preprocess_used = False
        self._auto_camera_preprocess_used = False
        self._feature_preprocess_runner_name = None
        self._camera_preprocess_runner_name = None
        self._camera_angular_velocity = None
        self._camera_motion_source = "none"
        self._camera_motion_provenance = {
            "source": "none",
            "verified": False,
            "runner": None,
            "asset": None,
        }
        self._dpvo_verified = False
        self._incompatible_assets = []
        self._camera_calibration = None
        self._dpvo_weights_available = False
        self._image_features_used = False
        self._feature_runner = None
        self._feature_runner_resolution_attempted = False
        self._feature_runner_error = None
        self._feature_runner_module = None
        self._feature_runner_checkpoint = None
        self._feature_runner_model_definition = None
        self._feature_runner_config_path = None
        self._feature_runner_active = False
        self._feature_runner_kind = None
        self._feature_error_code = None
        self._feature_next_action = None
        self._feature_runner_metadata = {}
        self._feature_fallback_reason = None
        self._feature_fallback_active = False
        self._feature_fallback_kind = None
        self._feature_fallback_consumed = False
        self._feature_adopted_frame_indices = []
        self._feature_input_frame_indices = []
        self._feature_source_provenance = (
            "verified_archive" if self._feature_source is not None else "none"
        )
        self._feature_source_verified = self._feature_source is not None
        self._trajectory_used = False
        self._trajectory_refiner_used = False
        self._smpl_output_used = False
        self._loaded_components = {
            "nativeMotionEncoder": False,
            "imageFeatureIntegrator": False,
            "trajectoryDecoder": False,
            "smplDecoder": False,
            "trajectoryRefiner": False,
        }
        try:
            if self._torch is None:
                raise RuntimeError("PyTorch is not installed in the selected Python environment")
            model_path = self._resolve_model_path()
            checkpoint = _safe_torch_load(self._torch, model_path, self.device)
            state = _extract_state_dict(checkpoint)
            self._architecture = _infer_architecture(state, self.config)
            model = NativeWhamNetwork(
                n_joints=self._architecture["n_joints"],
                in_dim=self._architecture["in_dim"],
                d_embed=self._architecture["d_embed"],
                rnn_type=self._architecture["rnn_type"],
                n_layers=self._architecture["n_layers"],
                d_feat=self._architecture["d_feat"],
                enable_decoders=self._full_decoder_requested,
            )
            target_state = model.state_dict()
            compatible = {}
            shape_mismatch = []
            for key, value in state.items():
                if key not in target_state:
                    continue
                value_shape = tuple(getattr(value, "shape", ()))
                if value_shape != tuple(target_state[key].shape):
                    shape_mismatch.append(
                        f"{key}: checkpoint {value_shape}, adapter {tuple(target_state[key].shape)}"
                    )
                    continue
                compatible[key] = value
            required_prefixes = ("motion_encoder.embed_layer.", "motion_encoder.regressor.")
            if not any(key.startswith(required_prefixes) for key in compatible):
                raise RuntimeError(
                    "checkpoint has no compatible WHAM motion_encoder weights; expected the official "
                    "wham_vit_w_3dpw.pth.tar state-dict"
                )
            result = model.load_state_dict(compatible, strict=False)
            missing_core = [
                key for key in result.missing_keys if key.startswith("motion_encoder.")
            ]
            if missing_core:
                raise RuntimeError(
                    "checkpoint is missing native WHAM motion_encoder tensors: "
                    + ", ".join(missing_core[:4])
                )
            self._loaded_parameter_count = len(compatible)
            component_prefixes = {
                name: prefix
                for name, prefix in {
                    "nativeMotionEncoder": "motion_encoder.",
                    "imageFeatureIntegrator": "integrator.",
                    "trajectoryDecoder": "trajectory_decoder.",
                    "smplDecoder": "motion_decoder.",
                    "trajectoryRefiner": "trajectory_refiner.",
                }.items()
            }
            for name, prefix in component_prefixes.items():
                expected = [key for key in target_state if key.startswith(prefix)]
                self._loaded_components[name] = bool(expected) and all(
                    key in compatible for key in expected
                )
            if shape_mismatch:
                self._optional_errors.append(
                    "checkpoint tensor shape mismatches omitted from native load: "
                    + "; ".join(shape_mismatch[:4])
                )
            # Build the optional body model before releasing checkpoint
            # references.  The official 527 MB archive embeds neutral SMPL
            # buffers, so this does not require a separate asset download.
            self._initialize_smpl_assets(state)
            del checkpoint
            del state
            del target_state
            del compatible
            self._model = model.to(self.device)
            self._model.eval()
            self.model_path = model_path
            # Resolve the image-side detector only after checkpoint
            # compatibility is confirmed.  A corrupt/incompatible archive
            # should fail fast without downloading or constructing a second
            # MediaPipe graph.
            self._initialize_detector()
            if self._require_smpl and self._smpl_model is None and self._smpl_seed is None:
                raise RuntimeError(
                    self._smpl_error or
                    "WHAM SMPL-derived initialization was requested but no body-model asset is available"
                )
            if self._strict_optional_assets:
                if self._require_image_features and not self._loaded_components["imageFeatureIntegrator"]:
                    raise RuntimeError(
                        "WHAM image-feature integration was requested but the checkpoint has no compatible integrator"
                    )
                if self._loaded_components["imageFeatureIntegrator"] and self._feature_source is None:
                    raise RuntimeError(
                        "WHAM strict optional-assets mode requires an offline image feature archive or extractor"
                    )
                if self._enable_trajectory and not self._loaded_components["trajectoryDecoder"]:
                    raise RuntimeError(
                        "WHAM trajectory decoding was requested but the checkpoint has no compatible decoder"
                    )
                if self._enable_trajectory and self._camera_source is None:
                    raise RuntimeError(
                        "WHAM strict optional-assets mode requires cameraPoses, slamResults, or cameraAngularVelocity"
                    )
                if self._loaded_components["smplDecoder"] and self._smpl_model is None:
                    raise RuntimeError(
                        self._smpl_error or
                        "WHAM strict optional-assets mode requires SMPL body-model buffers or smplModelPath"
                    )
            self._preflight_status = "model_loaded"
            return True
        except Exception as exc:
            self._error = str(exc)
            self._preflight_status = "failed"
            self._preflight_error = str(exc)
            self._model = None
            if self._detector is not None and self.config.get("detector") is None:
                close = getattr(self._detector, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                self._detector = None
            return False

    def _run_detector(
        self,
        frames_bgr: Sequence[np.ndarray],
        timestamps_sec: Sequence[float],
        frame_indices: Sequence[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        features: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        points2d: List[np.ndarray] = []
        scores: List[np.ndarray] = []
        for frame, timestamp, frame_index in zip(frames_bgr, timestamps_sec, frame_indices):
            height = int(frame.shape[0]) if frame is not None and frame.ndim >= 2 else 1
            width = int(frame.shape[1]) if frame is not None and frame.ndim >= 2 else 1
            observation = None
            if frame is not None and getattr(self._detector, "is_initialized", True):
                observation = self._detector.detect(
                    frame, timestamp_sec=float(timestamp), frame_index=int(frame_index)
                )
            points, point_scores = _as_keypoints2d(observation)
            feature, mask, _ = _normalize_frame(points, point_scores, width, height)
            features.append(feature)
            masks.append(mask)
            points2d.append(np.column_stack((points, point_scores)).astype(np.float32))
            scores.append(point_scores)
        if not features:
            return (
                np.zeros((0, 37), dtype=np.float32),
                np.zeros((0, 17), dtype=np.bool_),
                np.zeros((0, 17, 3), dtype=np.float32),
                np.zeros((0, 17), dtype=np.float32),
            )
        return (
            np.stack(features, axis=0),
            np.stack(masks, axis=0),
            np.stack(points2d, axis=0),
            np.stack(scores, axis=0),
        )

    def infer_sequence(
        self,
        frames_bgr: Sequence[np.ndarray],
        timestamps_sec: Sequence[float],
        frame_indices: Sequence[int],
    ) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError(self._error or "native WHAM adapter is not initialized")
        if frames_bgr is None or len(frames_bgr) == 0:
            return {"frames": [], "metadata": self.get_metadata()}
        if len(timestamps_sec) != len(frames_bgr) or len(frame_indices) != len(frames_bgr):
            raise ValueError("frames_bgr, timestamps_sec, and frame_indices must have equal lengths")
        try:
            normalized_indices = [int(value) for value in frame_indices]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"frame_indices are not integer-like: {exc}") from exc
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("frame_indices contain duplicate source frame IDs")
        if normalized_indices != list(frame_indices):
            raise ValueError("frame_indices must contain integer source frame IDs")
        frame_indices = normalized_indices
        total_f = len(frames_bgr)
        sys.stderr.write(f"[TexMotion] [1/4] Extracting 2D pose observations across {total_f} frames...\n")
        sys.stderr.flush()
        features, masks, points2d, scores = self._run_detector(
            frames_bgr, timestamps_sec, frame_indices
        )
        # Official WHAM preprocessing derives the HMR2 image crop from the
        # same tracked 2D observations used by the temporal core.  Run the
        # detector once, derive aligned center/scale boxes, and pass them
        # through the optional feature stage so an HMR2 runner cannot silently
        # encode a full frame or lose source-frame identity.
        frame_bboxes = _person_bboxes_from_keypoints(
            points2d[..., :2] if points2d.ndim == 3 else points2d,
            scores,
            frames_bgr,
        )
        sys.stderr.write("[TexMotion] [2/4] Preparing temporal features and camera models...\n")
        sys.stderr.flush()
        # Resolve optional full-pipeline stages before the regular camera and
        # image-feature loaders.  The generated archives are aligned to the
        # exact sampled frame indices supplied by the extractor.
        self._prepare_local_runtime(
            frames_bgr,
            timestamps_sec,
            frame_indices,
            frame_bboxes=frame_bboxes,
        )
        init_kp3d, init_smpl6d, _ = self._smpl_initialization()
        first_init = np.concatenate(
            (init_kp3d.reshape(-1), features[0]), axis=0
        )
        x = self._torch.from_numpy(features).unsqueeze(0).to(self.device)
        mask = self._torch.from_numpy(masks).unsqueeze(0).to(self.device)
        init = self._torch.from_numpy(first_init).unsqueeze(0).unsqueeze(1).to(self.device)
        camera_np = self._camera_for_sequence(len(frames_bgr))
        image_features_np = self._image_features_for_sequence(
            len(frames_bgr),
            frames_bgr,
            timestamps_sec,
            frame_indices,
            frame_bboxes,
        )
        if image_features_np is not None and not self._loaded_components["imageFeatureIntegrator"]:
            raise RuntimeError(
                "WHAM image features were supplied, but the checkpoint has no compatible "
                "official image-feature integrator"
            )

        trajectory: Optional[Dict[str, np.ndarray]] = None
        smpl_output: Optional[Dict[str, Any]] = None
        pred_pose = pred_shape = pred_cam = pred_contact = None
        device_label = str(self.device).upper()
        sys.stderr.write(f"[TexMotion] [3/4] Running WHAM recurrent sequence optimization on {device_label}...\n")
        sys.stderr.flush()
        with self._torch.inference_mode():
            prepared = self._model.preprocess(x, mask)
            pred_kp3d, motion_context = self._model.motion_encoder(prepared, init)
            motion_context_raw = motion_context
            if (
                self._enable_trajectory
                and self._loaded_components["trajectoryDecoder"]
            ):
                root_init = self._root_initialization(init_smpl6d)
                init_root = self._torch.from_numpy(root_init).to(self.device)
                cam_angvel = self._torch.from_numpy(camera_np).unsqueeze(0).to(self.device)
                pred_root, pred_vel = self._model.trajectory_decoder(
                    motion_context_raw,
                    init_root,
                    cam_angvel,
                )
                trajectory = self._trajectory_arrays(pred_root, pred_vel, camera_np)
                self._trajectory_used = True

            if image_features_np is not None:
                image_features = self._torch.from_numpy(image_features_np).unsqueeze(0).to(self.device)
                motion_context = self._model.integrator(motion_context, image_features)
                self._image_features_used = True

            if self._loaded_components["smplDecoder"]:
                sys.stderr.write("[TexMotion] [4/4] Decoding 3D humanoid motion and SMPL joint kinematics...\n")
                sys.stderr.flush()
                _, init_pose, _ = self._smpl_initialization()
                decoder_init = self._torch.from_numpy(init_pose).unsqueeze(0).unsqueeze(1).to(self.device)
                pred_pose, pred_shape, pred_cam, pred_contact = self._model.motion_decoder(
                    motion_context,
                    decoder_init,
                )
                smpl_output = self._decode_smpl(pred_pose, pred_shape)

            if (
                self._enable_trajectory_refiner
                and self._loaded_components["trajectoryRefiner"]
                and trajectory is not None
                and smpl_output is not None
                and pred_contact is not None
            ):
                refiner_output = {
                    "poses_root_r6d": self._torch.from_numpy(
                        trajectory["root_orientation_raw"]
                    ).unsqueeze(0).to(self.device),
                    "feet": smpl_output["feet"],
                    "contact": pred_contact,
                }
                cam_angvel = self._torch.from_numpy(camera_np).unsqueeze(0).to(self.device)
                refined = self._model.trajectory_refiner(
                    motion_context_raw,
                    self._torch.from_numpy(trajectory["root_velocity_raw"])
                    .unsqueeze(0)
                    .to(self.device),
                    refiner_output,
                    cam_angvel,
                )
                trajectory = self._trajectory_arrays(
                    refined["poses_root_r6d_refined"],
                    refined["vel_root_refined"],
                    camera_np,
                )
                self._trajectory_refiner_used = True
        # WHAM's released checkpoint is camera-space (Y-down, Z-away), while
        # the adapter contract is canonical SMPL-X (Y-up, Z-forward).  Convert
        # once here instead of making each caller guess the raw convention.
        predicted = _canonicalize_wham_coordinates(
            pred_kp3d.detach().cpu().numpy()[0]
        )

        metadata_snapshot = self.get_metadata()
        frames: List[Dict[str, Any]] = []
        for index in range(len(predicted)):
            frame_score = float(np.clip(np.mean(scores[index]), 0.0, 1.0))
            # Include a per-joint score column so the generic backend can
            # preserve visibility/prediction status while mapping the native
            # WHAM output to the canonical 33-slot MediaPipe topology.
            # ``keypoints2d`` and the motion encoder's learned 3D output are
            # both COCO-17 order: the first 17 rows of WHAM's embedded SMPL
            # regressor are nose/eyes/ears, shoulders, elbows, wrists, hips,
            # knees and ankles.  External adapters can still opt into the
            # historical ``wham_j17`` (common/H36M) mapping explicitly.
            score3d = np.clip(scores[index] * 0.8 + 0.2, 0.0, 1.0)
            landmarks3d = np.column_stack((predicted[index], score3d)).astype(np.float32)
            frame = {
                # Preserve source identity through the generic backend.  The
                # official temporal path may reorder/pad batched results, and
                # downstream fusion must never pair a pose with a different
                # video timestamp merely because list positions changed.
                "frameIndex": int(frame_indices[index]),
                "timestampSec": float(timestamps_sec[index]),
                "keypoints2d": points2d[index],
                "landmarks3d": landmarks3d,
                "confidence": frame_score,
                "metadata": dict(metadata_snapshot or {}),
            }
            if smpl_output is not None:
                for key in ("joints", "feet", "vertices", "offset"):
                    if key in smpl_output:
                        value = smpl_output[key][0, index]
                        frame["smpl" + key[0].upper() + key[1:]] = value.detach().cpu().numpy()
                self._smpl_output_used = True
            if pred_pose is not None:
                frame["smplPose6d"] = pred_pose[0, index].detach().cpu().numpy()
                frame["smplBetas"] = pred_shape[0, index].detach().cpu().numpy()
                frame["smplCamera"] = pred_cam[0, index].detach().cpu().numpy()
                frame["contact"] = pred_contact[0, index].detach().cpu().numpy()
            if trajectory is not None:
                frame["rootOrientation6d"] = trajectory["root_orientation"][index]
                frame["rootVelocity"] = trajectory["root_velocity"][index]
                frame["translation"] = trajectory["translation"][index]
                frame["cameraAngularVelocity"] = camera_np[index]
                frame["worldMotionAvailable"] = self._camera_motion_source in ("provided", "dpvo")
            frames.append(frame)
        result = {
            "frames": frames,
            "metadata": metadata_snapshot,
            "confidence": np.mean(scores, axis=1).astype(np.float32),
        }
        if trajectory is not None:
            result["trajectory"] = trajectory
        if smpl_output is not None:
            result["smpl"] = {
                key: value[0].detach().cpu().numpy()
                for key, value in smpl_output.items()
                if hasattr(value, "detach")
            }
        return result

    def _trajectory_arrays(
        self,
        root: "torch.Tensor",
        velocity: "torch.Tensor",
        camera_angular_velocity: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Convert official raw trajectory tensors to adapter conventions."""
        root_world, trans_world = _rollout_global_motion_torch(root, velocity)
        root_raw = root.detach().cpu().numpy()[0].astype(np.float32)
        velocity_raw = velocity.detach().cpu().numpy()[0].astype(np.float32)
        translation_raw = trans_world.detach().cpu().numpy()[0].astype(np.float32)
        yup_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        root_orientation = _matrix_to_rotation_6d_numpy(
            yup_flip @ _rotation_to_matrix_numpy(root_raw)
        )
        return {
            "root_orientation_raw": root_raw,
            "root_velocity_raw": velocity_raw,
            "root_orientation": root_orientation,
            "root_velocity": _canonicalize_wham_coordinates(velocity_raw),
            "translation": _canonicalize_wham_coordinates(translation_raw),
            "cameraAngularVelocity": camera_angular_velocity.astype(np.float32),
            "worldMotionAvailable": np.asarray(self._camera_motion_source in ("provided", "dpvo")),
        }

    def infer_frame(self, frame_bgr: np.ndarray, timestamp_sec: float, frame_index: int = 0) -> Any:
        result = self.infer_sequence([frame_bgr], [timestamp_sec], [frame_index])
        return result.get("frames", [None])[0]

    def _capabilities(self) -> Dict[str, bool]:
        smpl_seed_available = self._smpl_seed is not None
        smpl_derived = smpl_seed_available and self._smpl_seed[2] != "neutral_fallback"
        return {
            "nativeMotionEncoder": bool(self._loaded_components["nativeMotionEncoder"]),
            "imageFeatureIntegratorWeights": bool(self._loaded_components["imageFeatureIntegrator"]),
            "imageFeatureIntegrator": bool(
                self._loaded_components["imageFeatureIntegrator"] and self._image_features_used
            ),
            "imageFeatureBackbone": bool(self._feature_backbone_path),
            "imageFeatureArchive": bool(self._feature_source is not None),
            "trajectoryDecoderWeights": bool(self._loaded_components["trajectoryDecoder"]),
            "trajectoryDecoder": bool(self._trajectory_used),
            "smplDecoderWeights": bool(self._loaded_components["smplDecoder"]),
            "smplDecoder": bool(self._smpl_output_used),
            "smplInitialization": bool(smpl_seed_available),
            "smplDerivedInitialization": bool(smpl_derived),
            "embeddedSMPL": bool(isinstance(self._smpl_model, EmbeddedSMPL)),
            "cameraCalibration": bool(self._camera_calibration is not None),
            "cameraMotion": bool(self._camera_motion_source in ("provided", "dpvo", "configured_camera_pose")),
            "dpvoWeights": bool(self._dpvo_weights_available),
            "trajectoryRefinerWeights": bool(self._loaded_components["trajectoryRefiner"]),
            "trajectoryRefiner": bool(self._trajectory_refiner_used),
            "worldTrajectory": bool(
                self._trajectory_used and self._camera_motion_source in ("provided", "dpvo")
            ),
        }

    def _missing_optional_assets(self) -> List[str]:
        missing: List[str] = []
        capabilities = self._capabilities()
        if capabilities["imageFeatureIntegratorWeights"] and not self._image_features_used:
            missing.append(
                "image_feature_extractor_for_backbone"
                if self._feature_backbone_path and self._feature_source is None
                else "image_feature_archive_or_extractor"
            )
        if capabilities["smplDecoderWeights"] and self._smpl_model is None:
            missing.append("smpl_body_model_or_embedded_smpl_buffers")
        if capabilities["smplDecoderWeights"] and not capabilities["smplDerivedInitialization"]:
            missing.append("smpl_derived_first_frame_initialization")
        if self._camera_motion_source not in ("provided", "dpvo"):
            missing.append("camera_motion_or_slam")
        if self._smpl_error:
            missing.append("smpl_asset_error")
        return list(dict.fromkeys(missing))

    def get_metadata(self) -> Dict[str, Any]:
        capabilities = self._capabilities()
        missing = self._missing_optional_assets()
        optional_errors = list(dict.fromkeys(self._optional_errors))
        if self._smpl_error:
            optional_errors.append(self._smpl_error)
        if self._feature_fallback_reason:
            optional_errors.append(self._feature_fallback_reason)
        incompatible_assets = list(dict.fromkeys(
            str(item) for item in self._incompatible_assets if str(item).strip()
        ))
        feature_source = None
        if self._feature_fallback_reason:
            feature_source = "unverified_local_descriptor_rejected"
        elif self._image_features_used:
            feature_source = "archive" if isinstance(self._feature_source, (str, os.PathLike)) else "extractor"
        elif self._feature_backbone_path:
            feature_source = (
                "unverified_local_descriptor_rejected"
                if self._feature_fallback_reason else "backbone_only"
            )
        elif self._feature_source is not None:
            feature_source = "configured_not_used"
        feature_input_indices = [int(index) for index in self._feature_input_frame_indices]
        feature_adopted_indices = [int(index) for index in self._feature_adopted_frame_indices]
        feature_adopted_set = set(feature_adopted_indices)
        feature_rejected_indices = [
            index for index in feature_input_indices if index not in feature_adopted_set
        ]
        if self._image_features_used:
            feature_stage_status = "used"
        elif self._feature_fallback_reason or self._feature_runner_error:
            feature_stage_status = "rejected"
        elif self._feature_source is not None or self._feature_runner_active:
            feature_stage_status = "available"
        else:
            feature_stage_status = "unavailable"
        runtime_kind = self._feature_runner_metadata.get(
            "runtimeKind",
            self._feature_runner_metadata.get("runtime", "hmr2" if self._feature_runner_kind == "official_hmr2" else "vitpose"),
        )
        feature_runner_name = (
            self._feature_runner_metadata.get("runner")
            or self._feature_runner_kind
            or ("archive" if self._feature_source is not None else None)
        )
        feature_dim = self._feature_runner_metadata.get("featureDim")
        if feature_dim is None:
            feature_dim = self._architecture.get("d_feat")
        image_feature_stage = {
            "requestedRunner": "hmr2" if str(runtime_kind).lower() == "hmr2" else "vitpose",
            "runner": feature_runner_name,
            "runtimeKind": runtime_kind,
            "status": feature_stage_status,
            "runtimePath": self._feature_runner_metadata.get("runtimePath"),
            "checkpointPath": self._feature_runner_checkpoint or self._feature_backbone_path,
            "checkpointSha256": _sha256_file(self._feature_runner_checkpoint or self._feature_backbone_path),
            "featureDim": int(feature_dim) if feature_dim is not None else None,
            "frameCount": len(feature_input_indices),
            "acceptedFrameCount": len(feature_adopted_indices) if feature_stage_status != "unavailable" else 0,
            "modelFactory": self._feature_runner_metadata.get("modelFactory"),
            "errorCode": self._feature_error_code,
            "error": self._feature_runner_error or self._feature_fallback_reason,
            "nextAction": self._feature_next_action,
        }
        return {
            "adapter": "texmotion_wham_adapter",
            "adapterImplementation": "texmotion_native_wham",
            "selectedBackend": "wham",
            "backendReady": bool(self._preflight_status == "passed"),
            "preflightStatus": self._preflight_status,
            "preflightError": self._preflight_error,
            "nativeWhamCore": capabilities["nativeMotionEncoder"],
            "checkpoint": self.model_path,
            "modelPath": self.model_path,
            "device": self.device,
            "inputDetector": self._detector_name,
            "imageFeatureBackbonePath": self._feature_backbone_path,
            "imageFeatureSourceConfigured": bool(self._feature_source is not None),
            "imageFeatureSource": feature_source,
            "imageFeatureSourceProvenance": self._feature_source_provenance,
            "imageFeatureVerified": bool(self._feature_source_verified and self._image_features_used),
            "imageFeatureFallback": self._feature_fallback_kind if self._feature_fallback_active else None,
            "imageFeatureFallbackConsumed": bool(self._feature_fallback_consumed),
            "imageFeatureFallbackReason": self._feature_fallback_reason,
            "imageFeaturePreprocess": (
                self._feature_preprocess_runner_name if self._auto_feature_preprocess_used and feature_source == "archive"
                else None
            ),
            # Keep the inline runner decision explicit even when the local
            # descriptor fallback supplied the actual archive.  This is
            # consumed by the editor diagnostics and makes a missing/incompatible
            # model implementation distinguishable from a deliberate archive.
            "imageFeatureRunner": self._feature_runner_kind,
            "imageFeatureRunnerActive": bool(self._feature_runner_active),
            "imageFeatureRunnerStatus": (
                "active" if self._feature_runner_active
                else ("fallback" if (self._feature_runner_error or self._feature_fallback_reason) else "not_configured")
            ),
            "imageFeatureRunnerModule": self._feature_runner_module,
            "imageFeatureRunnerCheckpoint": self._feature_runner_checkpoint,
            "imageFeatureRunnerModelDefinition": self._feature_runner_model_definition,
            "imageFeatureRunnerConfigPath": self._feature_runner_config_path,
            "imageFeatureRunnerError": self._feature_runner_error,
            "imageFeatureRunnerRuntime": self._feature_runner_metadata.get("runtime"),
            "imageFeatureRunnerFeatureDim": self._feature_runner_metadata.get("featureDim"),
            "imageFeatureStage": image_feature_stage,
            "hmr2ImageFeaturesStatus": (
                "used" if feature_stage_status == "used" and str(runtime_kind).lower() == "hmr2"
                else feature_stage_status if str(runtime_kind).lower() == "hmr2" else "not_configured"
            ),
            "vitpose2d": {"status": "not_configured"},
            "vitpose2DStatus": "not_configured",
            "requestedBackend": self._requested_backend,
            "actualBackend": "wham",
            "backendFallback": False,
            "imageFeatureInputFrameIndices": feature_input_indices,
            "imageFeatureInputFrameCount": len(feature_input_indices),
            "imageFeatureExpectedFrames": len(feature_input_indices),
            "imageFeatureAdoptedFrames": feature_adopted_indices,
            "imageFeatureAdoptedFrameIndices": feature_adopted_indices,
            "imageFeatureAdoptedFrameCount": len(feature_adopted_indices),
            "imageFeatureRejectedFrames": feature_rejected_indices,
            "imageFeatureRunnerMetadata": dict(self._feature_runner_metadata),
            "whamPreprocessManifestPath": self._preprocess_manifest_path,
            "cameraAssetPath": self._camera_asset_path,
            "dpvoAssetPath": self._dpvo_asset_path,
            "cameraCalibrationAvailable": bool(self._camera_calibration is not None),
            "dpvoWeightsAvailable": bool(self._dpvo_weights_available),
            # The native motion encoder emits COCO-17 for both its 2D input
            # and learned 3D intermediate.  Keep the output topology explicit
            # so the generic backend applies the same semantic mapping; a
            # custom adapter may still advertise ``wham_j17`` when it emits
            # the common/H36M ordering.
            "inputJointTopology": "coco17",
            "jointTopology": "coco17",
            "outputJointTopology": "coco17",
            "coordinateSystem": "smplx",
            "rawCoordinateSystem": "wham_camera_y_down_z_away",
            "coordinateConversion": "raw (x,+Y down,+Z away) -> SMPL-X (x,+Y up,+Z forward)",
            "capabilities": capabilities,
            "capabilityFlags": capabilities.copy(),
            "missingOptionalAssets": missing,
            "missingAssets": list(missing),
            "incompatibleAssets": incompatible_assets,
            "optionalErrors": optional_errors,
            "whamInferencePolicy": "continue_with_explicit_fallback",
            "whamMissingAssetBehavior": (
                "Extraction continues when an optional WHAM asset or runner is missing; "
                "TexMotion never silently skips inference and records the exact fallback reason."
            ),
            "smplInitializationSource": self._smpl_seed[2] if self._smpl_seed is not None else None,
            "cameraMotionSource": self._camera_motion_source,
            "cameraMotionProvenance": dict(self._camera_motion_provenance),
            "dpvoVerified": bool(self._dpvo_verified),
            "cameraMotionRunner": (
                self._camera_preprocess_runner_name if self._auto_camera_preprocess_used and self._camera_source is not None
                else None
            ),
            "worldMotionAvailable": capabilities["worldTrajectory"],
            "cameraPreprocessSequenceSignature": self._camera_preprocess_signature,
            "officialImageFeatureIntegrator": capabilities["imageFeatureIntegrator"],
            "officialTrajectoryDecoder": capabilities["trajectoryDecoder"],
            "officialSmplDecoder": capabilities["smplDecoder"],
            "officialTrajectoryRefiner": capabilities["trajectoryRefiner"],
            "runtimeWarnings": list(dict.fromkeys(self._runtime_warnings)),
            "loadedParameterCount": int(self._loaded_parameter_count),
            "error": self._error,
        }

    def close(self) -> None:
        if self._feature_runner is not None:
            close_runner = getattr(self._feature_runner, "close", None)
            if callable(close_runner):
                try:
                    close_runner()
                except Exception:
                    pass
        if self._detector is not None:
            close = getattr(self._detector, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass
        self._detector = None
        self._feature_runner = None
        self._feature_runner_active = False
        self._model = None
        self._smpl_model = None


def create_backend(config: Optional[Dict[str, Any]] = None) -> NativeWHAMAdapter:
    return NativeWHAMAdapter(config)


__all__ = [
    "NativeWHAMAdapter",
    "NativeWhamNetwork",
    "EmbeddedSMPL",
    "Integrator",
    "TrajectoryDecoder",
    "MotionDecoder",
    "TrajectoryRefiner",
    "MotionEncoder",
    "NeuralInitialization",
    "Regressor",
    "_camera_angular_velocity_from_poses",
    "_coerce_smpl_initialization",
    "create_backend",
]
