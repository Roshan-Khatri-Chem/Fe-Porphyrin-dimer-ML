import numpy as np
import pytest
from geometry import monomer_frame, dimer_features

# ---------------------------------------------------------------------------
# Synthetic monomer builder
# ---------------------------------------------------------------------------

def make_monomer(center=np.zeros(3), R=np.eye(3)):
    """
    Build a minimal synthetic Fe-porphyrin in the xy-plane.

    Atom layout (in local frame before applying R):
      Fe  at (0, 0, 0)
      N1  at (+2.0,  0,   0)   <- reference nitrogen (smallest angle = 0 deg)
      N2  at (0,    +2.0, 0)
      N3  at (-2.0,  0,   0)
      N4  at (0,    -2.0, 0)
      8 C atoms at (+-2.8, +-1.1, 0) and (+-1.1, +-2.8, 0)
    """
    # Positions in local frame
    fe_pos = np.array([[0.0, 0.0, 0.0]])
    n_pos = np.array([
        [ 2.0,  0.0, 0.0],
        [ 0.0,  2.0, 0.0],
        [-2.0,  0.0, 0.0],
        [ 0.0, -2.0, 0.0],
    ])
    c_pos = np.array([
        [ 2.8,  1.1, 0.0], [-2.8,  1.1, 0.0],
        [ 2.8, -1.1, 0.0], [-2.8, -1.1, 0.0],
        [ 1.1,  2.8, 0.0], [-1.1,  2.8, 0.0],
        [ 1.1, -2.8, 0.0], [-1.1, -2.8, 0.0],
    ])
    positions = np.vstack([fe_pos, n_pos, c_pos])  # shape (13, 3)
    symbols = np.array(['Fe'] + ['N']*4 + ['C']*8)

    # Apply rotation then translation
    positions = (R @ positions.T).T + center
    return positions, symbols


def random_rotation():
    """Generate a random SO(3) rotation matrix via QR decomposition."""
    A = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_dimer(dz=3.5, dx=0.0, dy=0.0, phi=0.0, global_R=np.eye(3)):
    """
    Build a dimer. Monomer 1 is in the xy-plane at origin.
    Monomer 2 is translated by (dx, dy, dz) and rotated about z by phi.
    Then the whole dimer is rotated by global_R.
    """
    pos1, sym1 = make_monomer()

    # Rotation of monomer 2 about z axis by phi
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0,            0,           1],
    ])
    center2 = np.array([dx, dy, dz])
    pos2, sym2 = make_monomer(center=center2, R=Rz)

    # Apply global rotation
    pos1 = (global_R @ pos1.T).T
    pos2 = (global_R @ pos2.T).T

    return pos1, sym1, pos2, sym2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_T1_pure_stacking():
    """dz=3.5, no slippage, no twist -> expected features."""
    pos1, sym1, pos2, sym2 = make_dimer(dz=3.5)
    f = dimer_features(pos1, sym1, pos2, sym2)
    assert f.shape == (13,), "Feature vector must have shape (13,)"
    np.testing.assert_allclose(f[0], 3.5, atol=1e-6, err_msg="dz wrong")
    np.testing.assert_allclose(f[1], 0.0, atol=1e-6, err_msg="dx wrong")
    np.testing.assert_allclose(f[2], 0.0, atol=1e-6, err_msg="dy wrong")
    np.testing.assert_allclose(f[3], 1.0, atol=1e-6, err_msg="cos_theta wrong")
    np.testing.assert_allclose(f[4], 1.0, atol=1e-6, err_msg="cos_4phi wrong")
    np.testing.assert_allclose(f[5], 0.0, atol=1e-6, err_msg="sin_4phi wrong")


def test_T2_D4h_90deg_invariance():
    """
    KEY SYMMETRY TEST.
    Rotating monomer 2 by exactly 90 deg must give IDENTICAL features to phi=0.
    """
    f0   = dimer_features(*make_dimer(dz=3.5, phi=0.0))
    f90  = dimer_features(*make_dimer(dz=3.5, phi=np.pi/2))
    f180 = dimer_features(*make_dimer(dz=3.5, phi=np.pi))
    f270 = dimer_features(*make_dimer(dz=3.5, phi=3*np.pi/2))
    np.testing.assert_allclose(f0, f90,  atol=1e-10,
        err_msg="D4h VIOLATION: phi=0 and phi=90 deg give different features!")
    np.testing.assert_allclose(f0, f180, atol=1e-10,
        err_msg="D4h VIOLATION: phi=0 and phi=180 deg give different features!")
    np.testing.assert_allclose(f0, f270, atol=1e-10,
        err_msg="D4h VIOLATION: phi=0 and phi=270 deg give different features!")


def test_T3_D4h_45deg():
    """
    45 deg twist is the maximally different configuration under D4h.
    cos_4phi = cos(180 deg) = -1, sin_4phi = 0.
    """
    f = dimer_features(*make_dimer(dz=3.5, phi=np.pi/4))
    np.testing.assert_allclose(f[4], -1.0, atol=1e-10, err_msg="cos_4phi at 45 deg must be -1")
    np.testing.assert_allclose(f[5],  0.0, atol=1e-10, err_msg="sin_4phi at 45 deg must be 0")


def test_T4_global_rotation_invariance():
    """
    CRITICAL: Applying any rigid rotation to the WHOLE DIMER must not change
    any feature. Run 200 random rotations.
    """
    f_ref = dimer_features(*make_dimer(dz=3.8, dx=0.5, dy=0.3, phi=0.4))
    np.random.seed(0)
    for i in range(200):
        R = random_rotation()
        f_rot = dimer_features(*make_dimer(dz=3.8, dx=0.5, dy=0.3, phi=0.4, global_R=R))
        np.testing.assert_allclose(
            f_ref, f_rot, atol=1e-8,
            err_msg=f"ROTATION INVARIANCE VIOLATION at trial {i}: "
                    f"max diff = {np.max(np.abs(f_ref - f_rot)):.2e}"
        )


def test_T5_lateral_slippage():
    """dx=1.0 -> f[1] approx 1.0, dy=0 -> f[2] approx 0."""
    f = dimer_features(*make_dimer(dz=3.5, dx=1.0, dy=0.0))
    np.testing.assert_allclose(f[1], 1.0, atol=1e-6, err_msg="dx feature wrong")
    np.testing.assert_allclose(f[2], 0.0, atol=1e-6, err_msg="dy feature wrong")


def test_T6_no_nan_inf():
    """Feature vector must be finite for a variety of geometries."""
    configs = [
        dict(dz=3.5),
        dict(dz=3.5, dx=1.0, dy=0.8),
        dict(dz=5.0, phi=np.pi/4),
        dict(dz=3.5, phi=np.pi/8),
    ]
    for cfg in configs:
        f = dimer_features(*make_dimer(**cfg))
        assert np.all(np.isfinite(f)), f"NaN/Inf in features for config {cfg}: {f}"


def test_T7_dz_always_positive():
    """dz must be positive regardless of which monomer is labeled 1 or 2."""
    pos1, sym1, pos2, sym2 = make_dimer(dz=3.5)
    f_12 = dimer_features(pos1, sym1, pos2, sym2)
    f_21 = dimer_features(pos2, sym2, pos1, sym1)
    assert f_12[0] > 0, "dz must be positive (monomer order 1->2)"
    assert f_21[0] > 0, "dz must be positive (monomer order 2->1)"


def test_T8_feature_shape():
    """Output is exactly shape (13,) with dtype float64."""
    f = dimer_features(*make_dimer())
    assert f.shape == (13,)
    assert f.dtype == np.float64


def test_T9_d_lateral():
    """d_lateral = sqrt(dx^2 + dy^2), always >= 0."""
    # Pure stacking: d_lateral = 0
    f = dimer_features(*make_dimer(dz=3.5))
    np.testing.assert_allclose(f[8], 0.0, atol=1e-6, err_msg="d_lateral should be 0 for pure stacking")

    # dx=3, dy=4 -> d_lateral=5
    f = dimer_features(*make_dimer(dz=3.5, dx=3.0, dy=4.0))
    np.testing.assert_allclose(f[8], 5.0, atol=1e-6, err_msg="d_lateral should be 5.0")
    assert f[8] >= 0, "d_lateral must be non-negative"

    # dx only
    f = dimer_features(*make_dimer(dz=3.5, dx=2.0, dy=0.0))
    np.testing.assert_allclose(f[8], 2.0, atol=1e-6, err_msg="d_lateral should be 2.0")


def test_T10_d_lateral_rotation_invariance():
    """d_lateral and cos_4alpha/sin_4alpha must be rotation-invariant."""
    f_ref = dimer_features(*make_dimer(dz=3.8, dx=1.5, dy=0.8, phi=0.3))
    np.random.seed(42)
    for i in range(100):
        R = random_rotation()
        f_rot = dimer_features(*make_dimer(dz=3.8, dx=1.5, dy=0.8, phi=0.3, global_R=R))
        np.testing.assert_allclose(
            f_ref[8:], f_rot[8:], atol=1e-8,
            err_msg=f"New features not rotation-invariant at trial {i}"
        )


def test_T11_slippage_D4h_invariance():
    """
    Slippage direction cos_4alpha/sin_4alpha must be D4h-adapted:
    rotating the slippage direction by 90 degrees should give the same values.
    This is achieved because alpha -> alpha+90 gives 4*(alpha+90) = 4*alpha + 360.
    In the dimer, rotating the WHOLE dimer by 90 about the normal
    rotates the slippage direction by 90 (since x1 tracks monomer 1's frame).
    Actually, cos_4alpha and sin_4alpha should be invariant under 90-degree
    in-plane rotation of the slippage vector relative to the frame.
    """
    # dx=1, dy=0: alpha=0, cos_4alpha=1, sin_4alpha=0
    f = dimer_features(*make_dimer(dz=3.5, dx=1.0, dy=0.0))
    np.testing.assert_allclose(f[9], 1.0, atol=1e-6, err_msg="cos_4alpha wrong for alpha=0")
    np.testing.assert_allclose(f[10], 0.0, atol=1e-6, err_msg="sin_4alpha wrong for alpha=0")

    # dx=1, dy=1: alpha=45, cos_4alpha=cos(180)=-1
    f = dimer_features(*make_dimer(dz=3.5, dx=1.0, dy=1.0))
    np.testing.assert_allclose(f[9], -1.0, atol=1e-6, err_msg="cos_4alpha wrong for alpha=45")
    np.testing.assert_allclose(f[10], 0.0, atol=1e-6, err_msg="sin_4alpha wrong for alpha=45")


def test_T12_huckel_overlap_D4h_node():
    """
    |S_HL| must be zero when monomer 2 is rotated exactly 45 deg relative to
    monomer 1 (the D4h symmetry node). At this twist, every positive lobe
    of one monomer's HOMO aligns with a negative lobe of the other, causing
    exact cancellation.
    """
    f = dimer_features(*make_dimer(dz=3.5, phi=np.pi / 4))
    np.testing.assert_allclose(f[11], 0.0, atol=1e-10,
        err_msg="|S_HL| must be zero at phi=45 deg (D4h node)")
    np.testing.assert_allclose(f[12], 0.0, atol=1e-10,
        err_msg="S_HL^2 must be zero at phi=45 deg (D4h node)")

    # Also check 135, 225, 315 deg (all equivalent D4h nodes)
    for phi in [3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]:
        f = dimer_features(*make_dimer(dz=3.5, phi=phi))
        np.testing.assert_allclose(f[11], 0.0, atol=1e-10,
            err_msg=f"|S_HL| must be zero at phi={np.degrees(phi):.0f} deg")

    # |S_HL| should be NON-zero at phi=0 (eclipsed, maximum overlap)
    f0 = dimer_features(*make_dimer(dz=3.5, phi=0.0))
    assert f0[11] > 1e-3, "|S_HL| must be non-zero at phi=0 (eclipsed)"
    assert f0[12] > 1e-6, "S_HL^2 must be positive at phi=0"

    # D4h invariance: phi=0 and phi=90 must give same |S_HL|
    f90 = dimer_features(*make_dimer(dz=3.5, phi=np.pi / 2))
    np.testing.assert_allclose(f0[11], f90[11], atol=1e-10,
        err_msg="|S_HL| must be D4h invariant (phi=0 vs 90)")
    np.testing.assert_allclose(f0[12], f90[12], atol=1e-10,
        err_msg="S_HL^2 must be D4h invariant (phi=0 vs 90)")
