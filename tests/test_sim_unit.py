"""Fluid simulation unit tests: stability of both fish bodies, SimConfig
defaults, build_sims wiring. (test_sim.py covers the headless CLI run.)"""

import argparse

import numpy as np

import midline_fluid_sim as mfs


def run_steps(sim, steps=25):
    for _ in range(steps):
        sim.step()
    return sim


def test_simconfig_defaults_are_the_tuned_values():
    """Regression guard: the defaults must stay the stable tuned constants
    (the old spring-chain defaults rang violently and produced NaN)."""
    cfg = mfs.SimConfig()
    assert cfg.mass == 4.0
    assert cfg.k_bend == 5.0
    assert cfg.k_orient == 10.0
    assert cfg.k_act == 5.0
    assert cfg.damp == 0.03
    assert cfg.damp_v == 0.05
    assert cfg.cell_radius == 2.6
    assert cfg.c_drag == 0.03
    assert cfg.dt == 1.0
    assert cfg.max_ang_vel == 0.05
    assert cfg.max_node_speed == 0.18
    assert cfg.max_amp_freq == 0.03
    assert cfg.max_head_speed == 0.05
    assert cfg.force_scale == 3.0
    assert cfg.force_ema == 0.04
    assert cfg.torque_scale == 0.02


def test_rigid_chain_stays_finite():
    rest = mfs.build_rest_points(resolution=20, fish_length=80)
    joint_idx = list(range(0, 20, 4))
    sim = mfs.Simulation(120, 80, rest, joint_indices=joint_idx,
                         osc_amp=2.0, osc_freq=0.008)
    run_steps(sim)
    assert np.isfinite(sim.fish.pos).all()
    assert np.isfinite(sim.lb.f).all()
    assert np.abs(sim.fish.ang_vel).max() <= mfs.SimConfig().max_ang_vel + 1e-9


def test_rigid_chain_auto_joints():
    """Without joint_indices the body must still get evenly spaced joints."""
    rest = mfs.build_rest_points(resolution=30, fish_length=80)
    sim = mfs.Simulation(120, 80, rest, osc_amp=2.0, osc_freq=0.008)
    assert sim.fish.nseg >= 3
    run_steps(sim)
    assert np.isfinite(sim.fish.pos).all()


def test_rigid_chain_articulates_at_joints():
    """Regression guard: the rigid body must visibly bend at the joints rather
    than move as one stiff piece (old tuning was imperceptible)."""
    rest = mfs.build_rest_points(resolution=20, fish_length=80)
    joint_idx = list(range(0, 20, 4))
    sim = mfs.Simulation(120, 80, rest, joint_indices=joint_idx,
                         osc_amp=3.0, osc_freq=0.01)
    for _ in range(300):
        sim.step()
    dev = (sim.fish.angle - sim.fish.rest_angle)
    dev = (dev + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(dev).max() > 0.01, "rigid chain is not visibly articulating"
    assert np.isfinite(sim.fish.pos).all()


def test_lattice_has_no_walls():
    """Waves must pass the y edges (periodic) instead of reflecting back - a
    regression guard for the wall bounce-back removal. A pulse streaming past
    the top edge would be REFLECTED with bounce-back walls (f[4, 0, :] copied
    from the wrapped pulse, so f04 == f02); periodic boundaries keep the
    downward component at the low collision floor."""
    grid = mfs.LBGrid(16, 8)
    grid.f[2, 7, :] = 0.1  # upward-moving pulse at the bottom row
    grid.step(np.zeros((8, 16), dtype=bool), np.zeros((8, 16, 2)), np.zeros((8, 16, 2)))
    f02 = grid.f[2, 0, :].mean()  # wrapped pulse arriving at the top row
    f04 = grid.f[4, 0, :].mean()  # downward component at the top row
    assert f02 > 1e-9  # the pulse wrapped to the top row
    assert f04 < f02   # ...and was NOT reflected back downward


def test_simulation_progresses():
    rest = mfs.build_rest_points(resolution=16, fish_length=60)
    sim = mfs.Simulation(100, 60, rest, osc_amp=1.0, osc_freq=0.005)
    t0 = sim.sim_time()
    run_steps(sim, steps=10)
    assert sim.sim_time() > t0
    assert isinstance(sim.flow_speed(), float)
    # the oscillating head must actually move
    assert sim.fish.pos[0, 1] != sim.head0[1] or sim.fish.pos[0, 0] != sim.head0[0]


def test_flow_speed_converges_towards_target():
    rest = mfs.build_rest_points(resolution=12, fish_length=40, head=(35.0, 30.0))
    # nx must exceed 80 so the measurement window (40:nx-40) is non-empty
    sim = mfs.Simulation(140, 60, rest, target_speed=0.03, osc_amp=0.0, osc_freq=0.0)
    run_steps(sim, steps=60)
    # the body force drives a real current that stays finite near the target
    assert 0.0 < sim.flow_speed() < 0.08


def test_build_sims_single_and_sweep():
    args = argparse.Namespace(cycles=1.7, amplitude=22.0, length=100.0, nodes=20,
                              nx=120, ny=80, speed=0.035, drag=0.005,
                              amp=2.0, freq=0.008, axis="y", dt=0.05,
                              sweep=False, multipliers=[1.0])
    sims, names = mfs.build_sims(args)
    assert len(sims) == 1 and names == ["flow 0.035"]

    args.sweep = True
    args.multipliers = [1.0, 2.0, 4.0]
    sims, names = mfs.build_sims(args)
    assert len(sims) == 3 and len(names) == 3
    assert sims[1].target_speed == 0.07


def test_rasterisers_shared_and_correct():
    """_body_cells/_rasterize must agree with the class methods (dedupe guard)."""
    rest = mfs.build_rest_points(resolution=10, fish_length=50, head=(30.0, 30.0))
    fish = mfs.RigidChainFish(rest, [0, 5, 9], config=mfs.SimConfig())
    cells = fish.body_cells()
    assert cells, "no cells rasterised"
    solid, uw = fish.rasterize(100, 60)
    assert solid.shape == (60, 100) and uw.shape == (60, 100, 2)
    assert solid.any()


def test_body_force_zero_without_solid():
    """No solid cells -> the momentum-exchange force field is all zeros."""
    grid = mfs.LBGrid(16, 8)
    grid.step(np.zeros((8, 16), dtype=bool), np.zeros((8, 16, 2)), np.zeros((8, 16, 2)))
    assert np.allclose(grid.body_force, 0.0)


def test_body_force_drag_sign_in_current():
    """A +x current must push a stationary tethered body with a +x force
    (fluid drag downstream), so the momentum-exchange sign convention is
    physically correct."""
    rest = mfs.build_rest_points(amplitude=4.0, resolution=20, fish_length=50,
                                 head=(50.0, 60.0))
    sim = mfs.Simulation(200, 120, rest, joint_indices=[0, 5, 10, 15, 19],
                         target_speed=0.05, osc_amp=0.0, osc_freq=0.0)
    fx = []
    for _ in range(200):
        sim.step()
        fx.append(float(sim._node_force.sum(axis=0)[0]))
    assert np.mean(fx) > 0.01, "drag force on the fish must point downstream (+x)"
    assert np.isfinite(sim.lb.f).all()


def test_swim_mode_propels_forward_and_reports_metrics():
    """Free-swimming traveling wave: the fish must move forward (-x, head
    leading), stay finite and report the full metric set."""
    rest = mfs.build_rest_points(cycles=1.2, amplitude=4.0, resolution=24,
                                 fish_length=80, head=(60.0, 60.0))
    sim = mfs.Simulation(200, 120, rest, joint_indices=[0, 4, 8, 12, 16, 20, 23],
                         target_speed=0.0, osc_amp=2.5, osc_freq=0.014,
                         mode='swim')
    for _ in range(400):
        sim.step()
    assert np.isfinite(sim.fish.pos).all()
    assert np.isfinite(sim.lb.f).all()
    assert sim.fish.head[0] < sim.head0[0] - 0.5, "the fish must swim forward (head leading, -x)"
    m = sim.analyze()
    for key in ("speed", "speed_bl", "thrust", "power", "efficiency", "cost",
                "strouhal", "tail_amp", "track_rms", "yaw_rms", "wake",
                "finite", "body_len_cells"):
        assert key in m, key
    assert m["finite"] is True
    assert m["speed"] > 0.0
    assert m["thrust"] > 0.0
    assert m["tail_amp"] > 0.5


def test_swim_mode_deterministic():
    """Identical swim conditions must give identical metrics (no RNG)."""
    def run():
        rest = mfs.build_rest_points(cycles=1.2, amplitude=4.0, resolution=24,
                                     fish_length=80, head=(60.0, 60.0))
        sim = mfs.Simulation(200, 120, rest, joint_indices=[0, 4, 8, 12, 16, 20, 23],
                             target_speed=0.0, osc_amp=2.5, osc_freq=0.014,
                             mode='swim')
        for _ in range(250):
            sim.step()
        return sim.analyze()

    assert run() == run()


def test_collect_frames_includes_target_ghost_in_swim_mode():
    """collect_frames must record the commanded wave polyline per frame in
    swim mode and keep frames finite."""
    rest = mfs.build_rest_points(cycles=1.2, amplitude=6.0, resolution=24,
                                 fish_length=80, head=(60.0, 60.0))
    joint_idx = [0, 8, 16, 23]
    sim = mfs.Simulation(200, 120, rest, joint_indices=joint_idx,
                         target_speed=0.0, osc_amp=2.0, osc_freq=0.014,
                         mode='swim')
    frames, stride = mfs.collect_frames(sim, 60, joint_idx, frame_cap=15,
                                        include_target=True)
    assert 1 <= len(frames) <= 16  # the final step is always appended
    assert stride >= 1
    for fr in frames:
        assert "u" in fr and "body" in fr and "joints" in fr
        assert "target" in fr and len(fr["target"]) == len(fr["body"])
    assert sim.analyze()["steps"] == 60
