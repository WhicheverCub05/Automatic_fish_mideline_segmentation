"""
2D Lattice-Boltzmann fluid-structure simulation of an articulated fish body.

Two modes of movement:

- ``mode='tethered'`` (default): the head node is dragged along one axis with an
  oscillatory motion while a background water current (driven by a body force)
  flows through the domain - a flume test. The body responds to the mean flow
  through a linear drag law.
- ``mode='swim'``: FREE SWIMMING. The joints are actuated with a traveling wave
  of joint angles (exactly what a robotic fish's servos do) and the head is
  dynamic: the real fluid force on the body is measured from the LB momentum
  exchange (bounce-back) and accelerates the fish. The fish propels itself, and
  ``Simulation.analyze()`` reports swimming performance metrics (speed in
  cells/step and body lengths/s, thrust, actuation power, Froude efficiency,
  cost of transport, Strouhal number, tail-beat amplitude, wave-tracking RMS
  error, yaw RMS, wake speed). This is the mode used to TEST AND COMPARE joint
  configurations from the different generation algorithms.

The domain is fully periodic (no walls), so waves pass the edges instead of
reflecting - like a fish in a much larger pool. The fluid is simulated with a
D2Q9 lattice-Boltzmann solver where the body is a moving solid obstacle
(bounce-back); every LB step reports the momentum-exchange force on the solid
cells (``LBGrid.body_force``).

Usage:
    python midline_fluid_sim.py                          # single simulation, live animation
    python midline_fluid_sim.py --sweep                  # 3 flow speeds side by side
    python midline_fluid_sim.py --swim --headless --steps 500   # free-swim diagnostics + metrics
    python midline_fluid_sim.py --headless --steps 500   # run without animation, print diagnostics
"""

__author__ = "Alex R.d Silva"

import argparse
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from core import generate_midline_from_sinewave, is_interactive_backend

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])


@dataclass
class SimConfig:
    """
    Stability-critical constants for the articulated fish body, one place for
    every caller.

    The defaults are the TUNED values that are known-stable (validated through
    the web endpoint on tight, generated-segment node spacings). The CLI used to
    default to looser spring values that ring violently on sharp joint kinks
    (node velocities hit the clip and the moving bounce-back injected huge
    momentum -> NaN); SimConfig makes the stable values the single default.

    Tuned so the rigid segments visibly bend at the joints instead of moving as
    one stiff piece (light inertia, little damping, strong head coupling).
    """
    mass: float = 4.0  # segment mass (rod inertia = mass * seg_len^2 / 3)
    k_bend: float = 5.0  # bending torque per rad at a joint (tethered mode)
    k_orient: float = 10.0  # orientation torque per rad on the head segment
    k_act: float = 5.0  # servo stiffness per rad driving joints to the commanded wave
    damp: float = 0.03  # angular-velocity damping per step
    damp_v: float = 0.05  # head translational damping per step (swim mode)
    cell_radius: float = 2.6  # body rasterisation radius in lattice cells
    c_drag: float = 0.03  # flow drag coefficient on the body (tethered mode)
    dt: float = 1.0
    max_ang_vel: float = 0.05  # angular-velocity clip in RigidChainFish.integrate
    max_node_speed: float = 0.18  # combined node speed clip (swim mode, cells/step)
    max_amp_freq: float = 0.03  # actuation envelope: amp * freq must stay below this
    max_head_speed: float = 0.05  # head speed clip in swim mode (cells/step)
    force_scale: float = 3.0  # momentum-exchange force -> head acceleration scale
    force_ema: float = 0.04  # EMA smoothing of the per-node fluid forces (inertia)
    torque_scale: float = 0.02  # fluid-force torque on the segments (weak: the
    # body is motor-driven, so the fluid mainly translates the fish and only
    # gently resists the segments - the per-cell force noise would otherwise
    # overwhelm the servo torques and whip the body around)


def _body_cells(pos, cell_radius):
    """
    Rasterises node positions into lattice cells: every cell within cell_radius
    of a node belongs to that node. Shared by the fish body classes.
    :param pos: node positions (n, 2)
    :param cell_radius: cell radius of a node in lattice cells
    :return: dict mapping (x_index, y_index) to the owning node index
    """
    cells = {}
    r = cell_radius
    for k in range(len(pos)):
        cx, cy = pos[k]
        i0 = int(np.floor(cx - r))
        i1 = int(np.floor(cx + r)) + 1
        j0 = int(np.floor(cy - r))
        j1 = int(np.floor(cy + r)) + 1
        for i in range(i0, i1):
            for j in range(j0, j1):
                if (i - cx) ** 2 + (j - cy) ** 2 <= r * r:
                    cells[(i, j)] = k
    return cells


def _rasterize(cells, vel, nx, ny):
    """
    Builds the solid mask and wall velocities for the lattice from body cells.
    :param cells: dict mapping (x, y) to the owning node index (see _body_cells)
    :param vel: node velocities (n, 2)
    :param nx: lattice width
    :param ny: lattice height
    :return: solid boolean mask (ny, nx) and wall velocity field (ny, nx, 2)
    """
    solid = np.zeros((ny, nx), dtype=bool)
    uw = np.zeros((ny, nx, 2))
    for (i, j), k in cells.items():
        if 0 <= i < nx and 0 <= j < ny:
            solid[j, i] = True
            uw[j, i, 0] = vel[k, 0]
            uw[j, i, 1] = vel[k, 1]
    return solid, uw


class LBGrid:
    """
    D2Q9 lattice-Boltzmann solver with BGK collision, Guo forcing and a FULLY
    PERIODIC domain (no walls): the distribution streams wrap around in both
    x and y, so waves pass the edges instead of reflecting back - as if the
    fish were swimming in a much larger pool.
    """

    def __init__(self, nx, ny, tau=0.6):
        """Allocates the distribution arrays and sets the collision rate."""
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.omega = 1.0 / tau
        # distribution layout: (9, ny, nx) - the direction is the FIRST axis so
        # every per-direction operation works on a CONTIGUOUS plane (the old
        # (ny, nx, 9) layout made each f[:, :, d] a strided view, which costs
        # an extra copy on every roll / in-place op)
        self.f = np.zeros((9, ny, nx))
        self.f[0] = 1.0

    def velocity(self):
        """
        Macroscopic velocity and density from the distribution functions.
        :return: ux, uy, rho, all shaped (ny, nx)
        """
        f = self.f
        rho = f.sum(axis=0)
        rho = np.where(rho < 1e-12, 1e-12, rho)
        ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
        uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
        return ux, uy, rho

    def step(self, solid, uw, force):
        """
        Advances the fluid one time step using collide -> stream -> bounce-back.
        The domain is periodic in both axes (no walls). Solid body cells use
        moving-wall bounce-back so the flow cannot pass through the body.
        Also measures the momentum-exchange force ON the solid body (Ladd's
        method) and stores it in self.body_force shaped (ny, nx, 2).
        :param solid: boolean mask of the body cells, shaped (ny, nx)
        :param uw: velocity of the solid at each body cell, shaped (ny, nx, 2)
        :param force: external force field shaped (ny, nx, 2) applied to the fluid
        :return: None
        """
        f = self.f
        omega = self.omega
        Fx = force[:, :, 0]
        Fy = force[:, :, 1]

        rho = f.sum(axis=0)
        rho = np.where(rho < 1e-12, 1e-12, rho)
        ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7])
        uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8])
        ux = (ux + 0.5 * Fx) / rho
        uy = (uy + 0.5 * Fy) / rho

        # BGK collide + Guo forcing, then periodic streaming, direction by
        # direction (the planes are independent, and working plane-at-a-time
        # keeps the temporaries cache-sized)
        u2 = ux * ux + uy * uy
        uF = ux * Fx + uy * Fy
        f *= (1.0 - omega)
        for d in range(9):
            eu = CX[d] * ux + CY[d] * uy
            feq = W[d] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
            eF = CX[d] * Fx + CY[d] * Fy
            guo = (1.0 - 0.5 * omega) * W[d] * (3.0 * (eF - uF) + 9.0 * eu * eF)
            f[d] += omega * feq + guo
            if d:
                # periodic streaming in both axes (np.roll wraps the edges)
                f[d] = np.roll(f[d], (CY[d], CX[d]), axis=(0, 1))

        rho = f.sum(axis=0)
        rho = np.where(rho < 1e-12, 1e-12, rho)
        # momentum exchange force on the solid, computed from the INCOMING
        # populations BEFORE the bounce-back overwrites them (Ladd 1994):
        # force_on_solid = sum_d e_d * (2 * f_in_d - corr_d). Only fluid->solid
        # interface links exchange momentum with the body; interior solid-solid
        # links carry no physical force (their populations are internal noise).
        fx = np.zeros((self.ny, self.nx))
        fy = np.zeros((self.ny, self.nx))
        corrs = [None] * 9
        for d in range(1, 9):
            corr = 6.0 * W[d] * (CX[d] * uw[:, :, 0] + CY[d] * uw[:, :, 1]) * rho
            corrs[d] = corr
            m = 2.0 * f[d] - corr
            mask = solid & ~np.roll(solid, (CY[d], CX[d]), axis=(0, 1))
            fx += np.where(mask, CX[d] * m, 0.0)
            fy += np.where(mask, CY[d] * m, 0.0)
        self.body_force = np.stack([fx, fy], axis=-1)
        for d in range(1, 9):
            f[OPP[d]] = np.where(solid, f[d] - corrs[d], f[OPP[d]])


class _RasterizingBody:
    """
    Shared rasterisation behaviour for the articulated fish body: the current
    node positions become a solid mask plus moving-wall velocities for the
    lattice. Subclasses only need to provide `pos` (n, 2), `vel` (n, 2) and
    `cell_radius`.
    """

    def body_cells(self):
        """
        Rasterises the body into lattice cells: every cell within cell_radius of a
        node belongs to that node. Rebuilt each step as the body moves.
        :return: dict mapping (x_index, y_index) to the owning node index
        """
        return _body_cells(self.pos, self.cell_radius)

    def rasterize(self, nx, ny):
        """
        Builds the solid mask and wall velocities for the lattice from the current
        body position.
        :param nx: lattice width
        :param ny: lattice height
        :return: solid boolean mask (ny, nx) and wall velocity field (ny, nx, 2)
        """
        return _rasterize(self.body_cells(), self.vel, nx, ny)


class RigidChainFish(_RasterizingBody):
    """
    Articulated fish body of RIGID segments joined at joints - a robotic fish.
    There is no elasticity: segment lengths and interior node positions follow
    exactly from the segment angles, so the segments can never stretch or bend
    internally - the body only bends at the joints, driven by bending torques
    holding the generated joint angles. The state is the absolute angle of each
    segment; the head node is kinematic. Node velocities are the per-step
    displacements, which stay smooth, so the lattice wall velocities match the
    real body motion.
    """

    def __init__(self, rest_points, joint_indices=None, config=None):
        """
        Builds the rigid segments from the joint indices and rest shape. When
        joint_indices is None, joints are placed evenly along the body (every
        ~8 nodes) so the body still articulates without an explicit joint set.
        """
        cfg = self.config = config or SimConfig()
        self.rest = np.asarray(rest_points, dtype=float)
        self.n = len(self.rest)
        self.cell_radius = cfg.cell_radius
        self.dt = cfg.dt
        self.c_drag = cfg.c_drag
        self.damp = cfg.damp
        self.k_bend = cfg.k_bend
        self.k_orient = cfg.k_orient
        self.k_act = cfg.k_act
        self.u_mean = 0.0

        if joint_indices is None:
            nseg = max(3, self.n // 8)
            joint_indices = np.linspace(0, self.n - 1, nseg + 1)
        idx = sorted({int(i) for i in joint_indices if 0 <= int(i) < self.n})
        if len(idx) < 2:
            idx = [0, self.n - 1]
        if idx[0] != 0:
            idx.insert(0, 0)
        if idx[-1] != self.n - 1:
            idx.append(self.n - 1)
        self.joint_idx = idx
        self.nseg = len(idx) - 1

        self.seg_len = np.array([np.linalg.norm(self.rest[idx[a + 1]] - self.rest[idx[a]])
                                 for a in range(self.nseg)])
        self.rest_angle = np.array([np.arctan2(self.rest[idx[a + 1]][1] - self.rest[idx[a]][1],
                                               self.rest[idx[a + 1]][0] - self.rest[idx[a]][0])
                                    for a in range(self.nseg)])
        self.mass = np.full(self.nseg, cfg.mass)
        self.total_mass = float(self.mass.sum())
        # moment of inertia about the segment's near joint (rod)
        self.inertia = self.mass * self.seg_len ** 2 / 3.0
        self.inertia[self.inertia < 1e-9] = 1e-9

        self.head = self.rest[0].copy()
        self.prev_head = self.rest[0].copy()
        self.angle = self.rest_angle.copy()
        self.ang_vel = np.zeros(self.nseg)
        self.pos = self.rest.copy()
        self.vel = np.zeros((self.n, 2))
        self.hydro = np.zeros((self.nseg, 2))
        self.force_total = np.zeros(2)  # total fluid force on the body (swim mode)
        self.act_power = 0.0  # actuation work rate this step (swim mode)
        self.track_err = 0.0  # RMS joint-angle tracking error this step (swim mode)
        self._rebuild()

    def set_u_mean(self, u_mean):
        """
        Called by the simulation each step with the measured mean flow speed.
        :param u_mean: measured mean flow speed in lattice units
        :return: None
        """
        self.u_mean = u_mean

    def reset(self, head0):
        """
        Restores the body to its initial state.
        :param head0: head position to restore
        :return: None
        """
        self.head = np.asarray(head0, dtype=float).copy()
        self.prev_head = self.head.copy()
        self.angle = self.rest_angle.copy()
        self.ang_vel[:] = 0.0
        self.hydro[:] = 0.0
        self._rebuild()

    def set_head(self, pos):
        """
        Moves the head to the given position.
        :param pos: head position
        :return: None
        """
        pos = np.asarray(pos, dtype=float)
        self.head = pos.copy()
        self.prev_head = pos.copy()

    def _positions_from_angles(self, head, angles):
        """Derives node positions from absolute segment angles (exact, rigid)."""
        pos = np.zeros((self.n, 2))
        pos[0] = head
        for a in range(self.nseg):
            i0, i1 = self.joint_idx[a], self.joint_idx[a + 1]
            ux = np.cos(angles[a]) * self.seg_len[a]
            uy = np.sin(angles[a]) * self.seg_len[a]
            pos[i1] = pos[i0] + (ux, uy)
            n = i1 - i0
            for k in range(i0 + 1, i1):
                t = (k - i0) / n
                pos[k] = pos[i0] + (ux * t, uy * t)
        return pos

    def shape_from_angles(self, angles):
        """
        Node positions of the commanded shape (e.g. the target wave) anchored at
        the current head, used to draw the commanded-vs-actual ghost overlay.
        :param angles: absolute segment angles (radians)
        :return: array shaped (n, 2)
        """
        return self._positions_from_angles(self.head, np.asarray(angles, dtype=float))

    def _rebuild(self):
        """Derives all node positions from the segment angles (exact, rigid)."""
        pos = self._positions_from_angles(self.head, self.angle)
        self.vel = pos - self.pos  # node velocity from actual displacement
        self.pos = pos

    def integrate(self, target_angles=None, node_force=None):
        """
        Advances the body one time step.

        :param target_angles: commanded absolute segment angles (nseg,) for the
            traveling-wave servo mode; None keeps the tethered bending-spring
            behaviour around the rest angles.
        :param node_force: fluid force on each body node (n, 2) measured from the
            LB momentum exchange; None keeps the tethered mean-flow drag law.
        :return: None
        """
        T = np.zeros(self.nseg)
        self.act_power = 0.0
        self.track_err = 0.0
        if target_angles is not None:
            # servo mode: each joint motor drives its segment to the commanded
            # angle (a robotic fish); the actuation power is the servo work rate
            for a in range(self.nseg):
                T[a] += self.k_act * (target_angles[a] - self.angle[a])
                self.act_power += abs(self.k_act * (target_angles[a] - self.angle[a])
                                      * self.ang_vel[a])
            self.track_err = float(np.sqrt(np.mean((target_angles - self.angle) ** 2)))
        else:
            # bending springs at interior joints (relative angles)
            for a in range(1, self.nseg):
                dth = (self.angle[a] - self.angle[a - 1]) - (self.rest_angle[a] - self.rest_angle[a - 1])
                while dth > np.pi:
                    dth -= 2 * np.pi
                while dth < -np.pi:
                    dth += 2 * np.pi
                T[a - 1] += self.k_bend * dth
                T[a] -= self.k_bend * dth
        # orientation spring on the head segment (absolute angle; the swim mode
        # keeps the head segment straight, the tethered mode holds rest_angle[0])
        dth0 = self.angle[0] - (0.0 if target_angles is not None else self.rest_angle[0])
        while dth0 > np.pi:
            dth0 -= 2 * np.pi
        while dth0 < -np.pi:
            dth0 += 2 * np.pi
        T[0] += self.k_orient * (-dth0)
        if node_force is not None:
            # real fluid forces: total force (accelerates the head in swim mode)
            # and the torque of each node's force about its segment's near joint
            # (torque_scale keeps the motor-driven body stable against the
            # per-cell force noise)
            nf = np.asarray(node_force, dtype=float) * self.config.force_scale
            self.force_total = nf.sum(axis=0)
            ts = self.config.torque_scale
            for a in range(self.nseg):
                i0, i1 = self.joint_idx[a], self.joint_idx[a + 1]
                for k in range(i0 + 1, i1 + 1):
                    arm = self.pos[k] - self.pos[i0]
                    T[a] += ts * (arm[0] * nf[k, 1] - arm[1] * nf[k, 0])
        elif target_angles is None:
            # tethered mode only: hydro drag at the segment centres -> torque
            # about the near joint (swim mode uses the real fluid forces above)
            self.force_total = np.zeros(2)
            for a in range(self.nseg):
                i0, i1 = self.joint_idx[a], self.joint_idx[a + 1]
                v_center = (self.vel[i0] + self.vel[i1]) * 0.5
                fx = self.c_drag * (self.u_mean - v_center[0]) * self.seg_len[a]
                fy = self.c_drag * (0.0 - v_center[1]) * self.seg_len[a]
                ux = np.cos(self.angle[a])
                uy = np.sin(self.angle[a])
                half = self.seg_len[a] * 0.5
                T[a] += (ux * half) * fy - (uy * half) * fx
                self.hydro[a] = (fx, fy)
        else:
            # swim mode, first step: no measured fluid force yet
            self.force_total = np.zeros(2)
        self.ang_vel += (T / self.inertia) * self.dt
        self.ang_vel *= (1.0 - self.dt * self.damp)
        np.clip(self.ang_vel, -self.config.max_ang_vel, self.config.max_ang_vel, out=self.ang_vel)
        if target_angles is not None:
            # swim mode: bound the COMBINED node speed. Coherent rotation of all
            # segments swings the tail at sum(|ang_vel| * seg_len), which far
            # exceeds any per-segment limit and destabilises the bounce-back;
            # scaling the whole angular-velocity field keeps the wave shape
            # while capping every node velocity.
            spin = float(np.sum(np.abs(self.ang_vel) * self.seg_len))
            if spin > self.config.max_node_speed:
                self.ang_vel *= self.config.max_node_speed / spin
        self.angle += self.ang_vel * self.dt
        self._rebuild()


class Simulation:
    """
    Ties the fluid and the body together. Two modes:

    - 'tethered' (default): the head is dragged along one axis with an
      oscillatory motion and a background flow is driven by a body force.
    - 'swim': FREE SWIMMING - the joints are actuated with a traveling wave of
      joint angles (the servo model of a robotic fish) and the head is dynamic:
      the real fluid force on the body (LB momentum exchange) accelerates it.
      Per-step swimming statistics are recorded and analyze() reports the
      performance metrics used to compare joint configurations.
    """

    def __init__(self, nx, ny, rest_points, joint_indices=None, config=None, target_speed=0.035,
                 bulk_drag=0.005, sponge=0.01, flow_gain=0.003,
                 osc_amp=2.0, osc_freq=0.008, osc_axis='y', sim_dt=0.05, tau=0.6,
                 mode='tethered', cycles=1.0):
        """
        Creates the lattice, the articulated fish body, the flow controller and
        resets. In swim mode osc_amp/osc_freq are the tail-beat lateral
        amplitude (cells) and frequency (cycles/step) of the traveling wave;
        cycles is the number of wave cycles along the body.
        """
        self.nx = nx
        self.ny = ny
        self.mode = mode
        self.cycles = cycles
        self.lb = LBGrid(nx, ny, tau)
        self.fish = RigidChainFish(rest_points, joint_indices, config=config)
        self.config = self.fish.config
        self.bulk_drag = bulk_drag
        self.sponge = sponge
        self.flow_gain = flow_gain
        self.osc_amp = osc_amp
        self.osc_freq = osc_freq
        self.osc_axis = osc_axis
        self.sim_dt = sim_dt
        self.head0 = rest_points[0].copy()
        self.t = 0.0
        x = np.arange(nx)
        self.sponge_profile = np.clip(np.clip((10 - x) / 10, 0, 1) + np.clip((x - (nx - 10)) / 10, 0, 1), 0, 1)
        self._drag_profile = self.bulk_drag + self.sponge * self.sponge_profile[None, :]
        self._F = np.zeros((self.ny, self.nx, 2))
        self.set_flow_speed(target_speed)
        if mode == 'swim':
            # a robotic fish starts straight and bends into the wave, so the
            # swim baseline is a straight body (the config shape lives in the
            # joint placement, not in the initial pose)
            self.fish.angle[:] = 0.0
            self.fish.ang_vel[:] = 0.0
            self.fish._rebuild()
            self.fish.vel[:] = 0.0
        # swim-mode state
        self.head_vel = np.zeros(2)
        self.last_dx = 0.0
        self.last_target = None
        self._node_force = None
        self._seg_mid = np.zeros(self.fish.nseg)  # arc-length midpoints of the segments
        s = 0.0
        for a in range(self.fish.nseg):
            self._seg_mid[a] = s + self.fish.seg_len[a] * 0.5
            s += self.fish.seg_len[a]
        self._body_len = float(s)
        self.metrics = {k: [] for k in ("t", "speed", "thrust", "power", "tail_y",
                                        "yaw", "track", "flow")}
        self.reset()

    def set_flow_speed(self, speed):
        """
        Sets the target background flow speed (lattice cells per step). The body
        force that drives the flow is adjusted continuously to reach this speed.
        :param speed: target flow speed
        :return: None
        """
        self.target_speed = speed
        self.gx = speed * self.bulk_drag

    def sim_time(self):
        """
        Simulated time in seconds since the start of the simulation.
        :return: elapsed simulated time
        """
        return self.t * self.sim_dt

    def reset(self):
        """
        Restores the simulation to its initial state.
        :return: None
        """
        self.t = 0.0
        self.lb.f = np.zeros((9, self.ny, self.nx))
        self.lb.f[0] = 1.0
        self.fish.reset(self.head0)
        if self.mode == 'swim':
            self.fish.angle[:] = 0.0
            self.fish.ang_vel[:] = 0.0
            self.fish._rebuild()
            self.fish.vel[:] = 0.0
        self.head_vel[:] = 0.0
        self.last_dx = 0.0
        self.last_target = None
        self._node_force = None
        for k in self.metrics:
            self.metrics[k] = []

    def _wave_ramp(self):
        """
        Smoothly ramps the commanded wave in over the first two beat cycles so
        the initial kick cannot destabilise the moving bounce-back.
        :return: ramp factor in [0, 1]
        """
        ramp_steps = 2.0 / max(self.osc_freq, 1e-9)
        return min(1.0, self.t / ramp_steps)

    def _target_angles(self):
        """
        Commanded absolute segment angles of the traveling wave at time t:
        theta_a = env(s_a) * sin(2*pi*f*t - k*s_a), amplitude envelope growing
        linearly from the head to the tail. The envelope scale is solved
        analytically so the resulting TAIL LATERAL AMPLITUDE equals osc_amp.
        :return: array of commanded segment angles (nseg,)
        """
        k = 2.0 * np.pi * self.cycles / self._body_len
        theta = 2.0 * np.pi * self.cycles
        # amplitude of integral_0^L (s/L) sin(2pi f t - k s) ds, closed form
        z = (1j * theta + 1) * np.exp(-1j * theta) - 1
        f_theta = abs(z) / theta ** 2 if theta > 1e-9 else 0.5
        theta_tail = self.osc_amp / (self._body_len * f_theta)
        env = (self._seg_mid / self._body_len) * theta_tail
        return env * np.sin(2.0 * np.pi * self.osc_freq * self.t - k * self._seg_mid) * self._wave_ramp()

    def _gather_node_forces(self, cells):
        """
        Sums the LB momentum-exchange force on the body cells into per-node
        forces (n, 2), low-pass filtered over time (EMA). The instantaneous
        momentum exchange is far noisier than the physical force at these flow
        speeds, and feeding the spikes straight back into the body drives the
        moving wall faster than the bounce-back can handle; the EMA acts as the
        body's inertia and keeps the coupling stable.
        :param cells: dict mapping (x, y) cell to the owning node index
        :return: array shaped (n, 2)
        """
        fx, fy = self.lb.body_force[:, :, 0], self.lb.body_force[:, :, 1]
        node_force = np.zeros((self.fish.n, 2))
        for (i, j), k in cells.items():
            if 0 <= i < self.nx and 0 <= j < self.ny:
                node_force[k, 0] += fx[j, i]
                node_force[k, 1] += fy[j, i]
        if self._node_force is not None:
            a = self.config.force_ema
            node_force = a * node_force + (1.0 - a) * self._node_force
        if self.fish.n >= 5:
            # spatial smoothing along the body chain: the cell-level force is
            # far noisier than the physical distributed load, and a short box
            # filter keeps the smooth load while removing the per-cell spikes
            kernel = np.ones(3) / 3.0
            node_force[:, 0] = np.convolve(node_force[:, 0], kernel, mode='same')
            node_force[:, 1] = np.convolve(node_force[:, 1], kernel, mode='same')
        return node_force

    def step(self):
        """
        Advances the simulation one time step.
        :return: None
        """
        if self.mode == 'swim':
            # dynamic head: the fluid force measured at the previous LB step
            # accelerates the fish; damping + clip keep it stable
            v = self.head_vel
            v += (self.fish.force_total / self.fish.total_mass) * self.fish.dt
            v *= (1.0 - self.fish.dt * self.config.damp_v)
            np.clip(v, -self.config.max_head_speed, self.config.max_head_speed, out=v)
            head = self.fish.head + v * self.fish.dt
            self.last_dx = head[0] - self.fish.head[0]
            # keep the body inside the periodic domain (a safety net - default
            # runs stay far from the edges)
            head[0] %= self.nx
            head[1] %= self.ny
            self.fish.set_head(head)
            self.last_target = self._target_angles()
            self.fish.integrate(target_angles=self.last_target, node_force=self._node_force)
        else:
            if self.osc_axis == 'y':
                head = self.head0 + np.array([0.0, self.osc_amp * np.sin(2 * np.pi * self.osc_freq * self.t)])
            else:
                head = self.head0 + np.array([self.osc_amp * np.sin(2 * np.pi * self.osc_freq * self.t), 0.0])
            self.fish.set_head(head)
            self.fish.integrate()
        cells = self.fish.body_cells()
        solid, uw = _rasterize(cells, self.fish.vel, self.nx, self.ny)
        ux, uy, _ = self.lb.velocity()
        # no walls: the whole y range is open water (the x sponge edges stay
        # out of the measurement window); solid cells are excluded from the
        # flow measurement so the fish's own body does not pollute it
        window = ~solid[:, 40:self.nx - 40]
        measured = float(np.mean(ux[:, 40:self.nx - 40][window]))
        self.gx += self.flow_gain * (self.target_speed - measured)
        self.fish.set_u_mean(measured)
        # preallocated force field: F = gx - u * (bulk_drag + sponge profile)
        F = self._F
        F[:, :, 0] = self.gx - ux * self._drag_profile
        F[:, :, 1] = -uy * self._drag_profile
        self.lb.step(solid, uw, F)
        self._node_force = self._gather_node_forces(cells)
        if self.mode == 'swim':
            self.metrics['t'].append(self.t)
            self.metrics['speed'].append(-self.last_dx)  # forward = -x (head leading)
            self.metrics['thrust'].append(-self.fish.force_total[0])
            self.metrics['power'].append(self.fish.act_power)
            self.metrics['tail_y'].append(float(self.fish.pos[-1, 1]))
            self.metrics['yaw'].append(float(self.fish.angle[0]))
            self.metrics['track'].append(float(self.fish.track_err))
            self.metrics['flow'].append(measured)
        self.t += 1.0

    def analyze(self, warmup_frac=0.25):
        """
        Swimming performance metrics, averaged over the steps after the warm-up
        fraction (the transient where the fish accelerates from rest is
        excluded). Used to compare joint configurations on equal terms.
        :param warmup_frac: fraction of steps dropped at the start
        :return: dict of metrics (all floats, plus steps/warmup/finite)
        """
        m = self.metrics
        n = len(m['t'])
        if n < 4:
            return {'steps': n, 'warmup': 0, 'speed': 0.0, 'speed_bl': 0.0,
                    'thrust': 0.0, 'power': 0.0, 'efficiency': 0.0, 'cost': 0.0,
                    'strouhal': 0.0, 'tail_amp': 0.0, 'track_rms': 0.0,
                    'yaw_rms': 0.0, 'wake': 0.0, 'finite': True, 'body_len_cells': 0.0}
        w = int(n * warmup_frac)

        def win(key):
            return np.asarray(m[key][w:], dtype=float)

        speed = win('speed')
        thrust = win('thrust')
        power = win('power')
        tail = win('tail_y')
        yaw = win('yaw')
        track = win('track')
        flow = win('flow')
        u = float(speed.mean())
        fx = float(thrust.mean())
        p = float(power.mean())
        body_len = float(self.fish.seg_len.sum())
        tail_amp = float((tail.max() - tail.min()) / 2.0) if len(tail) else 0.0
        finite = bool(np.isfinite(self.lb.f).all() and np.isfinite(self.fish.pos).all())
        return {
            'steps': n,
            'warmup': w,
            'speed': round(u, 5),                      # mean forward speed, cells/step
            'speed_bl': round(u / body_len / self.sim_dt, 5) if body_len else 0.0,  # body lengths/s
            'thrust': round(fx, 5),                    # mean propulsive force, lattice units
            'power': round(p, 5),                      # mean actuation (servo) power
            'efficiency': round(fx * u / p, 5) if p > 1e-12 else 0.0,  # Froude efficiency
            'cost': round(p / u, 5) if abs(u) > 1e-9 else 0.0,         # power per distance
            'strouhal': round(self.osc_freq * 2.0 * tail_amp / u, 5) if abs(u) > 1e-9 else 0.0,
            'tail_amp': round(tail_amp, 4),            # measured tail-beat lateral amplitude
            'track_rms': round(float(np.sqrt(np.mean(track ** 2))), 5),  # wave tracking error
            'yaw_rms': round(float(np.sqrt(np.mean(yaw ** 2))), 5),      # head straightness
            'wake': round(float(np.mean(np.abs(flow))), 5),              # mean |flow| in window
            'finite': finite,
            'body_len_cells': round(body_len, 2),
        }

    def flow_speed(self):
        """
        Mean x velocity in the open water away from the fish.
        :return: measured flow speed in lattice units
        """
        ux, uy, _ = self.lb.velocity()
        return float(np.mean(ux[:, 40:self.nx - 40]))


def collect_frames(sim, steps, joint_idx, field_stride=None, frame_cap=600, include_target=False,
                   progress_cb=None):
    """
    Runs a simulation for `steps` steps and collects downsampled animation
    frames (velocity field, body polyline, joints, flow speed; plus the
    commanded wave ghost when include_target is set).
    :param sim: the Simulation to run
    :param steps: number of steps to run
    :param joint_idx: body node indices of the articulation joints
    :param field_stride: stride for the downsampled velocity field (default nx/48)
    :param frame_cap: maximum number of frames returned
    :param include_target: include the commanded wave polyline per frame (swim)
    :param progress_cb: optional progress_cb(fraction) called roughly every 1%
        of the steps (used by the web progress bar)
    :return: (frames, frame_stride) - frames is a list of dicts
    """
    if field_stride is None:
        field_stride = max(4, int(round(sim.nx / 48)))
    frame_stride = max(1, steps // frame_cap)
    report_every = max(1, steps // 100)
    frames = []
    for i in range(steps):
        sim.step()
        if progress_cb is not None and (i % report_every == 0 or i == steps - 1):
            progress_cb((i + 1) / steps)
        if i % frame_stride == 0 or i == steps - 1:
            ux, uy, _ = sim.lb.velocity()
            spd = np.sqrt(ux * ux + uy * uy)
            frame = {
                "u": np.round(spd[::field_stride, ::field_stride], 4).tolist(),
                "body": np.round(sim.fish.pos, 2).tolist(),
                "joints": np.round(sim.fish.pos[joint_idx], 2).tolist(),
                # flow speed from the SAME velocity field already computed
                # (saves a second full-grid pass per recorded frame)
                "flow": round(float(np.mean(ux[:, 40:sim.nx - 40])), 5),
            }
            if include_target and sim.last_target is not None:
                frame["target"] = np.round(sim.fish.shape_from_angles(sim.last_target), 2).tolist()
            frames.append(frame)
    return frames, frame_stride


def build_rest_points(cycles=1.7, amplitude=22.0, length_cm=110.0, resolution=40,
                      fish_length=100.0, head=(60.0, 80.0)):
    """
    Builds the body rest shape from a midline generated by the project's sine wave
    generator, scaled and positioned in the lattice domain.
    :param cycles: number of wave cycles
    :param amplitude: wave amplitude in cm
    :param length_cm: midline length in cm
    :param resolution: number of body nodes
    :param fish_length: body length in lattice cells
    :param head: head position in the lattice
    :return: rest points array shaped (resolution, 2)
    """
    midline = generate_midline_from_sinewave(cycles=cycles, amplitude=amplitude,
                                                length_cm=length_cm, phase_difference=0.0,
                                                frames=1, resolution=resolution)
    pts = np.array([[midline[p][0][0], midline[p][0][1]] for p in range(resolution)], dtype=float)
    scale = fish_length / length_cm
    pts[:, 0] *= scale
    pts[:, 1] *= scale
    pts[:, 0] += head[0] - pts[0, 0]
    pts[:, 1] += head[1] - pts[0, 1]
    return pts


def build_sims(args):
    """
    Builds the simulation(s) requested by the command line arguments.
    :param args: parsed command line arguments
    :return: list of simulations and their names
    """
    rest = build_rest_points(cycles=args.cycles, amplitude=args.amplitude,
                             resolution=args.nodes, fish_length=args.length,
                             head=(args.nx * 0.25, args.ny / 2))
    mode = getattr(args, 'swim', False) and 'swim' or 'tethered'
    wave_cycles = getattr(args, 'wave_cycles', 1.0)
    if args.sweep:
        sims = [Simulation(args.nx, args.ny, rest, target_speed=args.speed * m,
                           bulk_drag=args.drag, osc_amp=args.amp,
                           osc_freq=args.freq, osc_axis=args.axis, sim_dt=args.dt,
                           mode=mode, cycles=wave_cycles)
                for m in args.multipliers]
        names = [f"flow x{m}" for m in args.multipliers]
    else:
        sims = [Simulation(args.nx, args.ny, rest, target_speed=args.speed,
                           bulk_drag=args.drag, osc_amp=args.amp,
                           osc_freq=args.freq, osc_axis=args.axis, sim_dt=args.dt,
                           mode=mode, cycles=wave_cycles)]
        names = [f"flow {args.speed}"]
    return sims, names


def run_animation(sims, names, steps=2400, interval=30, slider=True):
    """
    Shows a live top-down animation of the simulation(s) with the flow field, the
    articulated body and the generated rest shape drawn on top.
    :param sims: list of simulations
    :param names: list of names for the subplots
    :param steps: number of steps per animation loop
    :param interval: milliseconds between frames
    :param slider: add a flow-force slider for a single simulation
    :return: None
    """
    if not is_interactive_backend():
        print("  non-interactive backend detected; showing a single frame.\n"
              "  install a GUI toolkit (e.g. tkinter) or run with MPLBACKEND=TkAgg to animate.")
        sims[0].step()
        ux, uy, _ = sims[0].lb.velocity()
        plt.imshow(np.sqrt(ux * ux + uy * uy), origin='lower', cmap='viridis')
        plt.plot(sims[0].fish.pos[:, 0], sims[0].fish.pos[:, 1], 'r-')
        plt.show()
        return

    fig, axes = plt.subplots(1, len(sims), figsize=(6.4 * len(sims), 4.8))
    axes = np.atleast_1d(axes)
    artists = []

    for ax, sim, name in zip(axes, sims, names):
        ux, uy, _ = sim.lb.velocity()
        spd = np.sqrt(ux * ux + uy * uy)
        im = ax.imshow(spd, origin='lower', cmap='viridis', vmin=0.0, vmax=0.05,
                       aspect='equal', extent=[0, sim.nx, 0, sim.ny])
        steps_x = np.arange(2, sim.nx, 8)
        steps_y = np.arange(2, sim.ny, 8)
        X, Y = np.meshgrid(steps_x, steps_y)
        qv = ax.quiver(X, Y, ux[Y, X], uy[Y, X], color='white', width=0.002, scale=25)
        body, = ax.plot([], [], 'r-', linewidth=2, label='rigid body')
        head, = ax.plot([], [], 'ro', markersize=7)
        rest, = ax.plot(sim.fish.rest[:, 0], sim.fish.rest[:, 1], 'w--', linewidth=1, alpha=0.7,
                        label='generated midline')
        ax.set_xlim(0, sim.nx)
        ax.set_ylim(0, sim.ny)
        ax.set_title(f"{name} - flow {sim.flow_speed():.4f} (target {sim.target_speed:.3f})")
        ax.legend(loc='upper right', fontsize=8)
        artists.append((im, qv, body, head, rest, steps_x, steps_y, ax, sim))

    def update(frame):
        """Animation step: resets at frame 0, advances every sim, redraws."""
        if frame == 0:
            state['t0'] = time.perf_counter()
            for s in sims:
                s.reset()
        for s in sims:
            s.step()
        elapsed = time.perf_counter() - state['t0']
        for im, qv, body, head, rest, steps_x, steps_y, ax, sim in artists:
            ux, uy, _ = sim.lb.velocity()
            X, Y = np.meshgrid(steps_x, steps_y)
            im.set_array(np.sqrt(ux * ux + uy * uy))
            qv.set_UVC(ux[Y, X], uy[Y, X])
            body.set_data(sim.fish.pos[:, 0], sim.fish.pos[:, 1])
            head.set_data([sim.fish.pos[0, 0]], [sim.fish.pos[0, 1]])
            ax.set_title(f"{name} - flow {sim.flow_speed():.4f} (target {sim.target_speed:.3f})")
        fig.suptitle(f"time {elapsed:.1f}s | sim {sims[0].sim_time():.1f}s - drag slider to change flow speed")
        return [im for im, *_ in artists]

    if slider and len(sims) == 1:
        from matplotlib.widgets import Slider
        ax_slider = plt.axes([0.18, 0.02, 0.64, 0.03])
        slider = Slider(ax_slider, 'flow speed (cells/step)', 0.0, 0.15,
                        valinit=sims[0].target_speed, valstep=0.001)

        def on_slider(val):
            """Live flow-force slider: updates the target speed of the sim."""
            sims[0].set_flow_speed(val)

        slider.on_changed(on_slider)
        slider_refs = [slider]  # noqa: F841 - keep the slider referenced so it is not GC'd

    state = {'t0': time.perf_counter()}
    ani = FuncAnimation(fig, update, frames=range(steps), interval=interval, blit=False, repeat=True)  # noqa: F841 - keep the animation referenced
    plt.show()


def headless_run(sims, steps):
    """
    Runs the simulation without animation, printing diagnostics, used for testing.
    :param sims: list of simulations
    :param steps: number of steps to run
    :return: None
    """
    sim = sims[0]
    for i in range(steps):
        sim.step()
        ux, uy, _ = sim.lb.velocity()
        max_u = float(np.max(np.sqrt(ux * ux + uy * uy)))
        if i % 50 == 0 or i == steps - 1:
            finite = bool(np.isfinite(sim.lb.f).all()) and bool(np.isfinite(sim.fish.pos).all())
            print(f"step {i:4d} | time {sim.sim_time():6.1f}s | flow {sim.flow_speed():.5f} | max|u| {max_u:.4f} | "
                  f"head ({sim.fish.pos[0, 0]:.1f},{sim.fish.pos[0, 1]:.1f}) | "
                  f"tail ({sim.fish.pos[-1, 0]:.1f},{sim.fish.pos[-1, 1]:.1f}) | finite {finite}")
    if sim.mode == 'swim':
        m = sim.analyze()
        print("swim metrics:")
        for k, v in m.items():
            print(f"  {k}: {v}")


def main():
    """Parses the CLI arguments and runs the animation or headless diagnostics."""
    parser = argparse.ArgumentParser(description="2D LBM fluid-structure simulation of an articulated fish body")
    parser.add_argument("--sweep", action="store_true", help="run 3 flow speeds side by side")
    parser.add_argument("--multipliers", nargs="+", type=float, default=[1.0, 2.0, 4.0],
                        help="flow speed multipliers for --sweep")
    parser.add_argument("--speed", type=float, default=0.035, help="target flow speed in cells/step")
    parser.add_argument("--drag", type=float, default=0.005, help="bulk drag on the flow")
    parser.add_argument("--amp", type=float, default=2.0, help="head oscillation amplitude in cells")
    parser.add_argument("--freq", type=float, default=0.008, help="head oscillation frequency (cycles/step)")
    parser.add_argument("--axis", choices=['y', 'x'], default='y', help="axis the head oscillates along")
    parser.add_argument("--dt", type=float, default=0.05, help="simulated seconds per time step")
    parser.add_argument("--cycles", type=float, default=1.7, help="sine wave cycles of the generated midline")
    parser.add_argument("--amplitude", type=float, default=22.0, help="sine wave amplitude in cm")
    parser.add_argument("--nodes", type=int, default=40, help="number of body nodes")
    parser.add_argument("--length", type=float, default=100.0, help="body length in lattice cells")
    parser.add_argument("--nx", type=int, default=260, help="lattice width")
    parser.add_argument("--ny", type=int, default=160, help="lattice height")
    parser.add_argument("--steps", type=int, default=2400, help="animation steps per loop")
    parser.add_argument("--interval", type=int, default=30, help="milliseconds between frames")
    parser.add_argument("--headless", action="store_true", help="run without animation and print diagnostics")
    parser.add_argument("--swim", action="store_true",
                        help="free-swimming mode: joints actuated with a traveling wave, "
                             "the fish propels itself and performance metrics are printed "
                             "(combine with --speed 0 for still water)")
    parser.add_argument("--wave-cycles", type=float, default=1.0,
                        help="wave cycles along the body in swim mode")
    args = parser.parse_args()

    sims, names = build_sims(args)
    if args.headless:
        headless_run(sims, args.steps)
    else:
        run_animation(sims, names, steps=args.steps, interval=args.interval)


if __name__ == "__main__":
    main()
