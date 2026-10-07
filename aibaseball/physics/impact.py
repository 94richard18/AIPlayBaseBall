"""Ball–bat collision: swept contact detection + 3D rigid-body impulse model (torch, batched).

Why analytic instead of PhysX contacts: real ball–bat contact lasts ~0.7 ms while the bat
sweet spot moves ~40 m/s, i.e. ~8 cm per 2 ms physics step (more than the ball diameter).
PhysX would tunnel or produce solver-dependent restitution. Here the bat is treated as a
free rigid body during the impulse (the standard "free bat" approximation: the hands do not
matter on the 1 ms impact time scale), with
  - speed-dependent ball COR and vibration loss away from the sweet spot (normal direction),
  - Coulomb friction / rolling condition (tangential direction) => realistic backspin from undercut.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .bat_model import BatSpec
from .constants import PhysicsSpec


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([z, -v[..., 2], v[..., 1]], -1),
            torch.stack([v[..., 2], z, -v[..., 0]], -1),
            torch.stack([-v[..., 1], v[..., 0], z], -1),
        ],
        -2,
    )


class BatGeometry:
    """Torch-side bat profile lookup and inertia."""

    def __init__(self, bat: BatSpec, device, dtype=torch.float32):
        xs, rs = zip(*bat.profile)
        self.xs = torch.tensor(xs, device=device, dtype=dtype)
        self.rs = torch.tensor(rs, device=device, dtype=dtype)
        props = bat.mass_properties()
        self.length = bat.length
        self.mass = bat.mass
        self.cm_x = props["cm_x"]
        self.i_axial = props["inertia_axial"]
        self.i_trans = props["inertia_transverse"]
        self.sweet_spot_x = bat.sweet_spot_x

    def radius_at(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, self.length)
        idx = torch.searchsorted(self.xs, x.contiguous()).clamp(1, len(self.xs) - 1)
        x0, x1 = self.xs[idx - 1], self.xs[idx]
        r0, r1 = self.rs[idx - 1], self.rs[idx]
        t = ((x - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
        return r0 + t * (r1 - r0)

    def inertia_world(self, axis: torch.Tensor) -> torch.Tensor:
        """I = It * Id + (Ia - It) * a a^T for an axisymmetric body with unit axis a."""
        eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(axis.shape[0], 3, 3)
        return self.i_trans * eye + (self.i_axial - self.i_trans) * axis.unsqueeze(-1) * axis.unsqueeze(-2)


@dataclass
class ContactQuery:
    hit: torch.Tensor  # (N,) bool
    frac: torch.Tensor  # (N,) sub-step fraction in [0,1] at first contact
    x_on_bat: torch.Tensor  # (N,) contact station along the bat [m from knob]
    normal: torch.Tensor  # (N,3) unit, from bat surface toward ball center
    point: torch.Tensor  # (N,3) contact point (world)


def closest_on_axis(ball_c, knob, axis, length):
    x = ((ball_c - knob) * axis).sum(-1).clamp(0.0, length)
    p = knob + x.unsqueeze(-1) * axis
    return x, p


@torch.no_grad()
def detect_contact(
    geom: BatGeometry,
    ball_c: torch.Tensor,
    ball_r: float,
    knob_prev: torch.Tensor,
    axis_prev: torch.Tensor,
    knob_now: torch.Tensor,
    axis_now: torch.Tensor,
    n_sub: int = 16,
    ball_prev: torch.Tensor | None = None,
) -> ContactQuery:
    """Swept test of the moving bat against the ball over one physics step.

    `ball_c` is the ball centre at the end of the step; pass `ball_prev` (start of the step) for a
    moving ball (pitched ball ~40 m/s = 10 cm per step) - both bodies are interpolated together.
    """
    # all sub-samples at once (S, N, 3): kernel count independent of n_sub, no host sync
    s = torch.linspace(0.0, 1.0, n_sub + 1, device=ball_c.device, dtype=ball_c.dtype).view(-1, 1, 1)
    knob = knob_prev + s * (knob_now - knob_prev)
    axis = torch.nn.functional.normalize(axis_prev + s * (axis_now - axis_prev), dim=-1)
    bc = ball_c.unsqueeze(0) if ball_prev is None else (ball_prev + s * (ball_c - ball_prev))
    x = ((bc - knob) * axis).sum(-1).clamp(0.0, geom.length)  # (S, N)
    d_vec = bc - (knob + x.unsqueeze(-1) * axis)
    dist = d_vec.norm(dim=-1)
    pen = dist - (geom.radius_at(x.flatten()).view_as(x) + ball_r)
    inside = pen <= 0.0
    hit = inside.any(dim=0)
    k = torch.argmax(inside.to(torch.int8), dim=0)  # first penetrating sample
    idx = torch.arange(ball_c.shape[0], device=ball_c.device)
    n_k = d_vec[k, idx] / dist[k, idx].clamp_min(1e-9).unsqueeze(-1)
    bc_k = bc.expand(s.shape[0], *ball_c.shape)[k, idx]  # ball centre at first contact
    return ContactQuery(hit=hit, frac=s.view(-1)[k], x_on_bat=x[k, idx], normal=n_k, point=bc_k - ball_r * n_k)


@dataclass
class ImpactResult:
    ball_vel: torch.Tensor  # (N,3) post-impact
    ball_omega: torch.Tensor  # (N,3) post-impact
    impulse_on_ball: torch.Tensor  # (N,3); bat receives -impulse at contact point
    cor: torch.Tensor  # (N,) effective COR used
    normal_speed: torch.Tensor  # (N,) approach speed along the normal (>0)


@torch.no_grad()
def resolve_impact(
    geom: BatGeometry,
    spec: PhysicsSpec,
    contact: ContactQuery,
    ball_c: torch.Tensor,
    ball_vel: torch.Tensor,
    ball_omega: torch.Tensor,
    bat_cm: torch.Tensor,
    bat_axis: torch.Tensor,
    bat_vel_cm: torch.Tensor,
    bat_omega: torch.Tensor,
) -> ImpactResult:
    ball, imp = spec.ball, spec.impact
    n = contact.normal
    p = contact.point
    r1 = p - ball_c  # ball center -> contact
    r2 = p - bat_cm  # bat CM -> contact

    v_rel = (ball_vel + torch.cross(ball_omega, r1, dim=-1)) - (bat_vel_cm + torch.cross(bat_omega, r2, dim=-1))
    vn = (v_rel * n).sum(-1)  # < 0 when approaching
    approach = (-vn).clamp_min(0.0)

    # effective COR: speed dependence (ball) x vibration loss (bat location)
    cor = imp.cor_ref + imp.cor_slope * (approach - imp.cor_ref_speed)
    cor = cor.clamp(imp.cor_min, imp.cor_max)
    dx = contact.x_on_bat - geom.sweet_spot_x
    cor = cor * (1.0 - imp.vib_loss_max * (1.0 - torch.exp(-((dx / imp.vib_width) ** 2))))

    # collision matrix K: Δv_rel(contact) = K J
    eye = torch.eye(3, device=n.device, dtype=n.dtype).expand(n.shape[0], 3, 3)
    i1_inv = 1.0 / ball.inertia
    i2_inv = torch.linalg.inv(geom.inertia_world(bat_axis))
    s1, s2 = _skew(r1), _skew(r2)
    K = (1.0 / ball.mass + 1.0 / geom.mass) * eye - i1_inv * (s1 @ s1) - s2 @ i2_inv @ s2

    v_t = v_rel - vn.unsqueeze(-1) * n
    # sticking solution: normal restitution + tangential velocity -> -e_t * v_t
    dv_target = (-(1.0 + cor) * vn).unsqueeze(-1) * n - (1.0 + imp.tangential_cor) * v_t
    J_stick = torch.linalg.solve(K, dv_target.unsqueeze(-1)).squeeze(-1)
    Jn_stick = (J_stick * n).sum(-1)
    Jt_stick = J_stick - Jn_stick.unsqueeze(-1) * n
    sliding = Jt_stick.norm(dim=-1) > imp.friction * Jn_stick.abs()

    # sliding solution: J = Jn (n + mu t), t opposes the initial slip
    t_hat = -torch.nn.functional.normalize(v_t, dim=-1, eps=1e-9)
    dir_slide = n + imp.friction * t_hat
    k_eff = (n * (K @ dir_slide.unsqueeze(-1)).squeeze(-1)).sum(-1)
    Jn_slide = -(1.0 + cor) * vn / k_eff.clamp_min(1e-9)
    J_slide = Jn_slide.unsqueeze(-1) * dir_slide

    J = torch.where(sliding.unsqueeze(-1), J_slide, J_stick)
    J = torch.where((vn < 0).unsqueeze(-1), J, torch.zeros_like(J))  # separating: no impulse

    new_v = ball_vel + J / ball.mass
    new_w = ball_omega + i1_inv * torch.cross(r1, J, dim=-1)
    return ImpactResult(ball_vel=new_v, ball_omega=new_w, impulse_on_ball=J, cor=cor, normal_speed=approach)
