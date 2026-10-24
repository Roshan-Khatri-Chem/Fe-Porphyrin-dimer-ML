import numpy as np
from ase.io import read as ase_read

# Node positions from Bakenhaster et al. Fig 4a at RFe-Fe = 5.0 Å
RPD_NODE_POSITIONS = [2.0, 4.6]   # Å
RPD_NODE_TOLERANCE = 0.5          # Å half-width — tuned in Step 3


def is_rpd_interference_node(d_lateral, tolerance=RPD_NODE_TOLERANCE):
    """
    Return True if d_lateral is within tolerance of a known
    lateral displacement interference zero.

    Node positions from Bakenhaster et al. Figure 4a:
      First zero:  RPD,X ≈ 2.0 Å
      Second zero: RPD,X ≈ 4.6 Å

    Note: these positions were measured at fixed RFe-Fe = 5.0 Å
    and may shift slightly with interlayer distance. The tolerance
    parameter accounts for this uncertainty.
    """
    for node in RPD_NODE_POSITIONS:
        if abs(d_lateral - node) < tolerance:
            return True
    return False


def load_dimer_xyz(filepath: str) -> dict:
    """
    Load a dimer XYZ file. Split into two monomers using sequential ordering:
    first N//2 atoms = monomer 1, second N//2 atoms = monomer 2.

    Returns
    -------
    dict with keys:
        'symbols'   : np.ndarray (N,)    dtype='<U2'  — all element symbols
        'positions' : np.ndarray (N, 3)  dtype=float64
        'monomer1'  : {'symbols': (M,), 'positions': (M, 3)}
        'monomer2'  : {'symbols': (M,), 'positions': (M, 3)}

    Raises
    ------
    ValueError  if N is odd
    ValueError  if monomer1 and monomer2 have different element compositions
                (sorted element lists must match)
    ValueError  if each monomer does not contain exactly 1 Fe atom
    """
    atoms = ase_read(filepath)
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    N = len(symbols)
    if N % 2 != 0:
        raise ValueError(f"{filepath}: odd atom count {N}")
    M = N // 2
    m1 = {'symbols': symbols[:M], 'positions': positions[:M]}
    m2 = {'symbols': symbols[M:], 'positions': positions[M:]}
    # Validate compositions match
    if sorted(m1['symbols'].tolist()) != sorted(m2['symbols'].tolist()):
        raise ValueError(f"{filepath}: monomers have different compositions")
    # Validate one Fe per monomer
    for i, m in enumerate([m1, m2], 1):
        fe_count = np.sum(m['symbols'] == 'Fe')
        if fe_count != 1:
            raise ValueError(f"{filepath}: monomer {i} has {fe_count} Fe atoms, expected 1")
    return {'symbols': symbols, 'positions': positions, 'monomer1': m1, 'monomer2': m2}


def monomer_frame(positions: np.ndarray, symbols: np.ndarray) -> tuple:
    """
    Compute a stable, reproducible right-handed orthonormal frame for one monomer.

    Parameters
    ----------
    positions : (M, 3) float64  — Cartesian coords of all monomer atoms
    symbols   : (M,)            — element symbols

    Returns
    -------
    centroid : (3,)   — mean position of the 4 porphyrin-core nitrogen atoms
    normal   : (3,)   — unit normal to the porphyrin plane (sign arbitrary here)
    x_axis   : (3,)   — unit vector in-plane toward the "reference nitrogen"
    """
    # STEP 1 — Identify the 4 porphyrin-core nitrogen atoms.
    # If more than 4 N atoms exist (e.g. Fe-TPP with amide linkages),
    # select the 4 closest to the Fe atom (the porphyrin core nitrogens).
    nitrogen_mask = (symbols == 'N')
    n_nitrogen = np.sum(nitrogen_mask)
    assert n_nitrogen >= 4, f"Expected >= 4 N atoms, got {n_nitrogen}"
    N_positions_all = positions[nitrogen_mask]
    if n_nitrogen > 4:
        fe_mask = (symbols == 'Fe')
        assert np.sum(fe_mask) >= 1, "Need at least 1 Fe atom"
        fe_pos = positions[fe_mask][0]
        dists = np.linalg.norm(N_positions_all - fe_pos, axis=1)
        core_idx = np.argsort(dists)[:4]
        N_positions = N_positions_all[core_idx]
    else:
        N_positions = N_positions_all  # (4, 3)

    # STEP 2 — Centroid
    centroid = N_positions.mean(axis=0)

    # STEP 3 — Normal via SVD
    N_centered = N_positions - centroid  # (4, 3)
    _, _, Vt = np.linalg.svd(N_centered)
    normal = Vt[-1]  # smallest singular value = plane normal

    # STEP 4 — x_axis: in-plane direction toward the first N atom in the array.
    # Using the first N atom (by array order) is intrinsic to the molecule's
    # data representation and therefore rotationally invariant.
    N_in_plane = N_positions - centroid
    x_axis_raw = N_in_plane[0]
    # Orthogonalize against normal (enforce in-plane)
    x_axis = x_axis_raw - np.dot(x_axis_raw, normal) * normal
    x_axis /= np.linalg.norm(x_axis)

    return centroid, normal, x_axis


def get_alpha_carbons(positions: np.ndarray, symbols: np.ndarray):
    """
    Identify the 8 alpha-carbon atoms (Cα bonded to N) and assign Hückel
    a2u HOMO coefficients with alternating sign pattern around the ring.

    The sign is determined intrinsically: each alpha-C is bonded to one N.
    The 4 N atoms are sorted by angle relative to the first N atom in array
    order (the same reference used by monomer_frame). The two alpha-C atoms
    bonded to each N are assigned the same sign as that N's pyrrole.
    The 4 pyrroles get alternating signs: +, -, +, -.

    Returns
    -------
    list of (position, coefficient) tuples, length 8.
    Coefficients are ±1/sqrt(8).
    """
    N_mask = (symbols == 'N')
    C_mask = (symbols == 'C')
    N_pos_all = positions[N_mask]
    C_idx = np.where(C_mask)[0]

    # Select 4 porphyrin-core N atoms (closest to Fe) if >4 N present
    if N_pos_all.shape[0] > 4:
        fe_mask = (symbols == 'Fe')
        fe_pos = positions[fe_mask][0]
        dists_fe = np.linalg.norm(N_pos_all - fe_pos, axis=1)
        core_idx = np.argsort(dists_fe)[:4]
        N_pos = N_pos_all[core_idx]
    else:
        N_pos = N_pos_all

    centroid = N_pos.mean(axis=0)

    # Alpha-carbons: C atoms within bonding distance (< 1.6 Å) of any core N
    alpha_idx = []
    alpha_bonded_N = []  # which N index (0-3) each alpha-C is bonded to
    for ci in C_idx:
        dists_to_N = np.linalg.norm(positions[ci] - N_pos, axis=1)
        if np.any(dists_to_N < 1.6):
            alpha_idx.append(ci)
            alpha_bonded_N.append(np.argmin(dists_to_N))

    assert len(alpha_idx) == 8, f"Expected 8 alpha-carbons, got {len(alpha_idx)}"

    alpha_pos = positions[alpha_idx]
    alpha_bonded_N = np.array(alpha_bonded_N)

    # Sort the 4 N atoms by angle around the centroid.
    # Fix the SVD normal sign using the cross product of two ADJACENT N atoms
    # (sorted by angle). Array-order N[0] and N[1] may be opposite in the ring,
    # making their cross product unstable. Sorting first gives adjacent pairs.
    _, _, Vt = np.linalg.svd(N_pos - centroid)
    normal = Vt[-1]
    N_in_plane = N_pos - centroid

    # Preliminary sort by angle using an arbitrary but deterministic in-plane axis
    x_pre = N_in_plane[0] - np.dot(N_in_plane[0], normal) * normal
    x_pre /= np.linalg.norm(x_pre)
    y_pre = np.cross(normal, x_pre)
    pre_angles = np.arctan2(N_in_plane @ y_pre, N_in_plane @ x_pre)
    pre_order = np.argsort(pre_angles)

    # Fix normal sign: cross product of two adjacent sorted N vectors should
    # have positive dot with normal. Adjacent atoms span ~90°, giving a robust
    # cross product.
    v0 = N_in_plane[pre_order[0]]
    v1 = N_in_plane[pre_order[1]]
    cross_adj = np.cross(v0, v1)
    if np.dot(cross_adj, normal) < 0:
        normal = -normal

    # Re-sort with the fixed normal
    x_ref = N_in_plane[0] - np.dot(N_in_plane[0], normal) * normal
    x_ref /= np.linalg.norm(x_ref)
    y_ref = np.cross(normal, x_ref)

    N_angles = np.arctan2(N_in_plane @ y_ref, N_in_plane @ x_ref)
    N_order = np.argsort(N_angles)  # N indices sorted by angle (counterclockwise)

    # Assign alternating pyrrole signs: +, -, +, -
    # N_order[0] gets +, N_order[1] gets -, etc.
    pyrrole_sign = np.zeros(4)
    for k, ni in enumerate(N_order):
        pyrrole_sign[ni] = 1.0 if k % 2 == 0 else -1.0

    # Each alpha-C gets the sign of its bonded N's pyrrole
    c0 = 1.0 / np.sqrt(8.0)
    result = []
    for k in range(8):
        sign = pyrrole_sign[alpha_bonded_N[k]]
        result.append((alpha_pos[k], sign * c0))

    return result


def huckel_overlap(pos1, sym1, pos2, sym2, alpha=0.5):
    """
    Compute approximate Hückel HOMO-HOMO overlap between two monomers.

    Uses Gaussian-decayed orbital products between the 8 alpha-carbons
    of each monomer, weighted by a2u HOMO coefficients.

    Parameters
    ----------
    pos1, sym1 : monomer 1 positions and symbols
    pos2, sym2 : monomer 2 positions and symbols
    alpha      : Gaussian decay parameter in Å^{-2}

    Returns
    -------
    S_HL : float, signed Hückel overlap
    """
    ac1 = get_alpha_carbons(pos1, sym1)
    ac2 = get_alpha_carbons(pos2, sym2)
    S = 0.0
    for r1, c1 in ac1:
        for r2, c2 in ac2:
            r2_dist = np.sum((r1 - r2) ** 2)
            S += c1 * c2 * np.exp(-alpha * r2_dist)
    return S


def dimer_features(m1_pos: np.ndarray, m1_sym: np.ndarray,
                   m2_pos: np.ndarray, m2_sym: np.ndarray) -> np.ndarray:
    """
    Compute the 13-dimensional symmetry-adapted feature vector for a dimer.

    Returns
    -------
    features : np.ndarray shape (13,) float64

    Features (in order):
        [0] dz          — interplanar spacing along ring-1 normal (Å), always > 0
        [1] dx          — lateral slippage along x1 axis (Å)
        [2] dy          — lateral slippage along y1 = n1 × x1 axis (Å)
        [3] cos_theta   — dot product of the two ring normals; +1 = parallel face-to-face
        [4] cos_4phi    — in-plane twist angle, cosine of 4φ (D4h symmetry-adapted)
        [5] sin_4phi    — in-plane twist angle, sine of 4φ (D4h symmetry-adapted)
        [6] cos_4psi    — tilt direction, cosine of 4ψ (D4h symmetry-adapted)
        [7] sin_4psi    — tilt direction, sine of 4ψ (D4h symmetry-adapted)
        [8] d_lateral   — sqrt(dx² + dy²), unsigned slippage magnitude (Å), >= 0
        [9] cos_4alpha  — slippage direction, cosine of 4·atan2(dy, dx) (D4h-adapted)
        [10] sin_4alpha — slippage direction, sine of 4·atan2(dy, dx) (D4h-adapted)
        [11] |S_HL|     — |Hückel HOMO-HOMO overlap| (unsigned, gauge-invariant)
        [12] S_HL_sq    — S_HL² (coupling ∝ J²)
    """
    # --- Frames ---
    c1, n1_raw, x1 = monomer_frame(m1_pos, m1_sym)
    c2, n2_raw, x2 = monomer_frame(m2_pos, m2_sym)

    # --- Orient normals ---
    # n1 points from ring 1 toward ring 2
    r_vec = c2 - c1
    n1 = n1_raw * np.sign(np.dot(n1_raw, r_vec))
    # n2 oriented same as n1 (along r_vec); face-to-face parallel → n1·n2 = +1
    n2 = n2_raw * np.sign(np.dot(n2_raw, r_vec))
    y1 = np.cross(n1, x1)  # completes the right-handed frame

    # --- Translational features ---
    dz = np.dot(r_vec, n1)  # always positive by construction
    dx = np.dot(r_vec, x1)
    dy = np.dot(r_vec, y1)

    # --- cos_theta ---
    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)

    # --- In-plane twist φ ---
    x2_proj = x2 - np.dot(x2, n1) * n1
    norm_x2_proj = np.linalg.norm(x2_proj)
    if norm_x2_proj > 1e-6:
        x2_proj /= norm_x2_proj
        cos_phi = np.clip(np.dot(x2_proj, x1), -1.0, 1.0)
        sin_phi = np.dot(x2_proj, y1)
        phi = np.arctan2(sin_phi, cos_phi)
    else:
        phi = 0.0
    cos_4phi = np.cos(4.0 * phi)
    sin_4phi = np.sin(4.0 * phi)

    # --- Tilt direction ψ ---
    tilt_axis = np.cross(n1, n2)
    tilt_mag = np.linalg.norm(tilt_axis)
    if tilt_mag > 1e-6:
        tilt_axis /= tilt_mag
        cos_psi = np.dot(tilt_axis, x1)
        sin_psi = np.dot(tilt_axis, y1)
        psi = np.arctan2(sin_psi, cos_psi)
    else:
        psi = 0.0
    cos_4psi = np.cos(4.0 * psi)
    sin_4psi = np.sin(4.0 * psi)

    # --- Lateral slippage magnitude and direction ---
    d_lateral = np.sqrt(dx**2 + dy**2)
    if d_lateral > 1e-6:
        alpha = np.arctan2(dy, dx)
    else:
        alpha = 0.0
    cos_4alpha = np.cos(4.0 * alpha)
    sin_4alpha = np.sin(4.0 * alpha)

    # --- Hückel HOMO overlap ---
    # Use |S_HL| rather than signed S_HL: the overall sign of a MO is a
    # gauge choice, so the signed overlap is not invariant under global
    # rotations. |S_HL| and S_HL² are both gauge-invariant.
    S_HL = huckel_overlap(m1_pos, m1_sym, m2_pos, m2_sym)
    abs_S_HL = abs(S_HL)
    S_HL_sq = S_HL ** 2

    return np.array([dz, dx, dy, cos_theta, cos_4phi, sin_4phi, cos_4psi, sin_4psi,
                     d_lateral, cos_4alpha, sin_4alpha, abs_S_HL, S_HL_sq],
                    dtype=np.float64)


def features_from_xyz(filepath: str) -> np.ndarray:
    """Convenience wrapper: filepath → (8,) feature vector."""
    d = load_dimer_xyz(filepath)
    return dimer_features(
        d['monomer1']['positions'], d['monomer1']['symbols'],
        d['monomer2']['positions'], d['monomer2']['symbols']
    )
