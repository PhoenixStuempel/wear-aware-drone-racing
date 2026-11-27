"""Batched quaternion helpers. Quaternions are (..., 4) as w, x, y, z and map body to world."""

import torch


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # v' = v + 2w (u x v) + 2 u x (u x v), u is the vector part. Cheaper than q v q*
    w, u = q[..., :1], q[..., 1:]
    t = 2 * torch.linalg.cross(u, v)
    return v + w * t + torch.linalg.cross(u, t)


def to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True)
    half = angle.unsqueeze(-1) / 2
    return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)
