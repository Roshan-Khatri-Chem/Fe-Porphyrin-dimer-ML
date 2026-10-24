"""
Rotation-aware S_12 computation using SRSH-PCM monomer HOMO.

For each dimer, we:
  1. Kabsch-align the reference porphine monomer onto each dimer monomer (m1, m2)
     to obtain rotation R_i and translation T_i, idealising the geometry.
  2. Build a PySCF dimer with the reference monomer atoms placed at R_i @ ref + T_i.
  3. Apply Wigner-D AO rotation for each angular-momentum shell of the reference
     HOMO to obtain c_i = R_AO(R_i) @ c_ref in PySCF spherical basis.
  4. Compute S_12 = c_1^T S_dimer c_2 (cross block of the dimer overlap matrix).

This gives a S_12 that respects the orbital character of the porphine HOMO under
arbitrary monomer rotations — important for MD-sampled geometries.
"""
import numpy as np
from pyscf import gto, symm
from scipy.spatial.transform import Rotation


def extract_porphine_core(positions, symbols):
    """Identify the 25 atoms of the porphine core (1 Fe + 4 N + 20 C) in a
    monomer that may have additional substituents (peptoid linkers, phenyl
    groups, etc.).

    Returns indices into the input arrays in the order:
      [Fe, N1, N2, N3, N4, alpha-C1..C8, beta-C1..C8, meso-C1..C4]

    Algorithm:
      1. Fe = the (single) Fe atom.
      2. 4 core N = the 4 N atoms closest to Fe.
      3. 8 alpha-C = C atoms within 1.6 A of any core N (each pyrrole has 2).
      4. 8 beta-C = C atoms within 1.6 A of any alpha-C, NOT a meso bridge.
      5. 4 meso-C = C atoms bonded to two different alpha-C atoms (bridge
         carbons connecting two pyrroles).
    """
    pos = np.asarray(positions)
    sym = np.asarray(symbols)
    fe_idx = np.where(sym == 'Fe')[0]
    if len(fe_idx) != 1:
        raise ValueError(f"Expected 1 Fe atom, got {len(fe_idx)}")
    fe = fe_idx[0]
    fe_pos = pos[fe]

    # 4 core N (closest to Fe)
    N_idx = np.where(sym == 'N')[0]
    d_to_fe = np.linalg.norm(pos[N_idx] - fe_pos, axis=1)
    core_N = N_idx[np.argsort(d_to_fe)[:4]]

    # alpha-C: C atoms within 1.6 A of any core N
    C_idx = np.where(sym == 'C')[0]
    alpha_C = []
    alpha_to_N = {}
    for ci in C_idx:
        d_to_N = np.linalg.norm(pos[ci] - pos[core_N], axis=1)
        if np.min(d_to_N) < 1.6:
            alpha_C.append(ci)
            alpha_to_N[ci] = core_N[np.argmin(d_to_N)]
    alpha_C = np.array(alpha_C, dtype=int)
    if len(alpha_C) != 8:
        raise ValueError(f"Expected 8 alpha-C atoms, got {len(alpha_C)}")

    # meso-C: C atoms bonded to TWO different alpha-C atoms (in different pyrroles)
    meso_C = []
    beta_C = []
    for ci in C_idx:
        if ci in alpha_C: continue
        d_to_alpha = np.linalg.norm(pos[ci] - pos[alpha_C], axis=1)
        bonded_alphas = alpha_C[d_to_alpha < 1.6]
        if len(bonded_alphas) == 0:
            continue  # not in porphine core
        bonded_pyrroles = set(alpha_to_N[a] for a in bonded_alphas)
        if len(bonded_pyrroles) >= 2:
            # bonded to alpha-Cs from different pyrroles -> meso-C
            meso_C.append(ci)
        elif len(bonded_pyrroles) == 1:
            # within same pyrrole -> beta-C
            beta_C.append(ci)
    meso_C = np.array(meso_C, dtype=int)
    beta_C = np.array(beta_C, dtype=int)
    if len(meso_C) != 4:
        raise ValueError(f"Expected 4 meso-C atoms, got {len(meso_C)}")
    if len(beta_C) != 8:
        raise ValueError(f"Expected 8 beta-C atoms, got {len(beta_C)}")

    # Concatenate in a stable order
    return np.concatenate([[fe], core_N, alpha_C, beta_C, meso_C])


def kabsch_align(ref_coords, target_coords):
    """Kabsch: find (R, T) minimising || R @ ref + T - target ||_F.

    R is 3x3 proper rotation, T is 3-vector translation.
    """
    ref = np.asarray(ref_coords)
    tgt = np.asarray(target_coords)
    assert ref.shape == tgt.shape
    cR = ref.mean(axis=0)
    cT = tgt.mean(axis=0)
    H = (ref - cR).T @ (tgt - cT)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    T = cT - R @ cR
    return R, T


def build_cart2sph_matrix(mol_cart, mol_sph):
    """Per-shell cart->sph projection (n_sph, n_cart). cart d/f are normalised='sp'."""
    n_cart = mol_cart.nao
    n_sph = mol_sph.nao
    U = np.zeros((n_sph, n_cart))
    cart_off = 0
    sph_off = 0
    for shell_id in range(mol_cart.nbas):
        l = mol_cart.bas_angular(shell_id)
        nctr = mol_cart.bas_nctr(shell_id)
        nc = (l + 1) * (l + 2) // 2
        ns = 2 * l + 1
        T = np.eye(nc) if l <= 1 else gto.cart2sph(l, normalized='sp')
        for _ in range(nctr):
            U[sph_off:sph_off + ns, cart_off:cart_off + nc] = T.T
            cart_off += nc
            sph_off += ns
    return U


def build_ao_rotation_matrix(mol_sph, R):
    """For a rigid rotation R applied to the molecule, build the (n_ao, n_ao)
    rotation matrix M such that c_rotated = M @ c (in spherical basis)."""
    euler = Rotation.from_matrix(R).as_euler('zyz', degrees=False)
    n_ao = mol_sph.nao
    M = np.zeros((n_ao, n_ao))
    off = 0
    for shell_id in range(mol_sph.nbas):
        l = mol_sph.bas_angular(shell_id)
        nctr = mol_sph.bas_nctr(shell_id)
        ns = 2 * l + 1
        D = symm.Dmatrix.Dmatrix(l, *euler, reorder_p=True)
        for _ in range(nctr):
            M[off:off + ns, off:off + ns] = D
            off += ns
    return M


class S12Calculator:
    """Computes S_12 between two monomer placements using a precomputed reference
    HOMO (in PySCF spherical basis) and the SRSH-PCM 6-31G(d) basis."""

    def __init__(self, basis_dict, ref_atom_list_ref_orient, homo_coeffs_sph,
                 ref_heavy_indices=None, ref_symbols=None, ref_coords=None):
        """
        Parameters
        ----------
        basis_dict : dict[str, list]
            PySCF-style basis dict (per element).
        ref_atom_list_ref_orient : list of (sym, [x,y,z])
            Reference monomer atom list (used for Kabsch reference; 37 atoms).
        homo_coeffs_sph : ndarray
            Reference HOMO coefficients in PySCF spherical basis (length n_ao_per_monomer).
        ref_heavy_indices : list[int] or None
            Indices of heavy atoms in the reference (for Kabsch alignment using
            heavy atoms only — robust against H positions).  If None, all atoms.
        """
        self.basis = basis_dict
        self.ref_atoms = ref_atom_list_ref_orient
        self.ref_coords = np.array([p for _, p in ref_atom_list_ref_orient])
        self.ref_syms = [s for s, _ in ref_atom_list_ref_orient]
        if ref_heavy_indices is None:
            self.heavy_indices = [i for i, s in enumerate(self.ref_syms) if s != 'H']
        else:
            self.heavy_indices = ref_heavy_indices

        # Build a single-monomer mol to know n_ao and to use for AO rotation
        self.mol_ref_sph = gto.M(atom=self.ref_atoms, basis=self.basis,
                                 unit='Angstrom', charge=0, spin=0, verbose=0, cart=False)
        self.n_ao_mono = self.mol_ref_sph.nao
        assert homo_coeffs_sph.shape == (self.n_ao_mono,), \
            f"HOMO coeffs shape {homo_coeffs_sph.shape} != ({self.n_ao_mono},)"
        self.c_ref = homo_coeffs_sph

    def _alignment(self, target_pos, target_sym):
        """Kabsch-align reference porphine core (Fe + 4 N) to target's porphine
        core. Uses only the 5-atom Fe+4N subset; this is robust to substituents
        (Fe-TPP, peptoid linkers) and to N labelling ambiguity (D4 symmetry).

        Returns
        -------
        R : 3x3 rotation
        T : 3-vector translation
        """
        ref_pos = np.asarray(self.ref_coords)
        ref_sym = np.asarray(self.ref_syms)
        ref_fe = ref_pos[ref_sym == 'Fe'][0]
        ref_N_all = ref_pos[ref_sym == 'N']
        # Reference porphyrin: 4 N in the porphine; closest to Fe
        d_ref = np.linalg.norm(ref_N_all - ref_fe, axis=1)
        ref_N = ref_N_all[np.argsort(d_ref)[:4]]

        tgt_pos = np.asarray(target_pos)
        tgt_sym = np.asarray(target_sym)
        tgt_fe = tgt_pos[tgt_sym == 'Fe'][0]
        tgt_N_all = tgt_pos[tgt_sym == 'N']
        d_tgt = np.linalg.norm(tgt_N_all - tgt_fe, axis=1)
        tgt_N = tgt_N_all[np.argsort(d_tgt)[:4]]

        # Sort both by angle around Fe using the porphine plane normal
        def sort_N_by_angle(N_pos, fe_pos):
            centered = N_pos - fe_pos
            _, _, Vt = np.linalg.svd(centered)
            normal = Vt[-1]
            x_pre = centered[0] - np.dot(centered[0], normal) * normal
            x_pre /= np.linalg.norm(x_pre)
            y_pre = np.cross(normal, x_pre)
            angles = np.arctan2(centered @ y_pre, centered @ x_pre)
            return N_pos[np.argsort(angles)]

        ref_N_sorted = sort_N_by_angle(ref_N, ref_fe)
        tgt_N_sorted = sort_N_by_angle(tgt_N, tgt_fe)

        # Try 4 cyclic permutations (and reflections) of the N labelling, pick best
        best_R, best_T, best_rmsd = None, None, np.inf
        for k in range(4):
            for reflect in [False, True]:
                if reflect:
                    # reverse cyclic order (handles the C4 vs S4 ambiguity)
                    perm = np.array([(k - i) % 4 for i in range(4)])
                else:
                    perm = np.array([(k + i) % 4 for i in range(4)])
                ref_5 = np.vstack([[ref_fe], ref_N_sorted])
                tgt_5 = np.vstack([[tgt_fe], tgt_N_sorted[perm]])
                R, T = kabsch_align(ref_5, tgt_5)
                rmsd = np.sqrt(np.mean(np.sum((ref_5 @ R.T + T - tgt_5) ** 2, axis=1)))
                if rmsd < best_rmsd:
                    best_rmsd = rmsd; best_R, best_T = R, T
        return best_R, best_T

    def compute(self, m1_pos, m1_sym, m2_pos, m2_sym):
        """Compute the rotation-aware S_12 for one dimer."""
        R1, T1 = self._alignment(m1_pos, m1_sym)
        R2, T2 = self._alignment(m2_pos, m2_sym)

        # Place reference monomer at idealised m1 and m2 positions
        ref_at_m1 = (self.ref_coords @ R1.T + T1).tolist()
        ref_at_m2 = (self.ref_coords @ R2.T + T2).tolist()
        atoms = [(s, p) for s, p in zip(self.ref_syms, ref_at_m1)] + \
                [(s, p) for s, p in zip(self.ref_syms, ref_at_m2)]
        mol_d = gto.M(atom=atoms, basis=self.basis, unit='Angstrom',
                      charge=0, spin=0, verbose=0, cart=False)
        S_d = mol_d.intor('int1e_ovlp')

        # Apply Wigner-D rotation to the reference HOMO coefficients
        M1 = build_ao_rotation_matrix(self.mol_ref_sph, R1)
        M2 = build_ao_rotation_matrix(self.mol_ref_sph, R2)
        c1_rot = M1 @ self.c_ref
        c2_rot = M2 @ self.c_ref

        # Insert into dimer coefficient vector
        nao = self.n_ao_mono
        c1 = np.zeros(mol_d.nao)
        c2 = np.zeros(mol_d.nao)
        c1[:nao] = c1_rot
        c2[nao:nao + nao] = c2_rot
        return float(c1 @ S_d @ c2)

    def compute_with_overlap_matrix(self, m1_pos, m1_sym, m2_pos, m2_sym):
        """Same as compute() but also returns the cross block of the AO overlap."""
        R1, T1 = self._alignment(m1_pos, m1_sym)
        R2, T2 = self._alignment(m2_pos, m2_sym)
        ref_at_m1 = (self.ref_coords @ R1.T + T1).tolist()
        ref_at_m2 = (self.ref_coords @ R2.T + T2).tolist()
        atoms = [(s, p) for s, p in zip(self.ref_syms, ref_at_m1)] + \
                [(s, p) for s, p in zip(self.ref_syms, ref_at_m2)]
        mol_d = gto.M(atom=atoms, basis=self.basis, unit='Angstrom',
                      charge=0, spin=0, verbose=0, cart=False)
        S_d = mol_d.intor('int1e_ovlp')
        nao = self.n_ao_mono
        S_cross = S_d[:nao, nao:nao + nao]
        M1 = build_ao_rotation_matrix(self.mol_ref_sph, R1)
        M2 = build_ao_rotation_matrix(self.mol_ref_sph, R2)
        c1_rot = M1 @ self.c_ref
        c2_rot = M2 @ self.c_ref
        return float(c1_rot @ S_cross @ c2_rot), S_cross


def load_srsh_homo_sph(fchk_path, normalize_to_pyscf_basis_fn,
                       reorder_mo_coeffs_to_pyscf_fn, build_pyscf_basis_fn,
                       parse_fchk_fn, periodic):
    """Load SRSH HOMO from a fchk file, return PySCF spherical-basis coefficient
    vector + the basis dict + atom list (in reference orientation).
    """
    fchk = parse_fchk_fn(fchk_path)
    basis, _ = build_pyscf_basis_fn(fchk)
    syms = [periodic[z] for z in fchk['atomnos']]
    ref_coords = fchk['coords_ang']
    atom_list = [(s, p.tolist()) for s, p in zip(syms, ref_coords)]

    mol_cart = gto.M(atom=atom_list, basis=basis, unit='Angstrom',
                     charge=0, spin=0, verbose=0, cart=True)
    mol_sph = gto.M(atom=atom_list, basis=basis, unit='Angstrom',
                    charge=0, spin=0, verbose=0, cart=False)

    S_cart = mol_cart.intor('int1e_ovlp')
    S_sph = mol_sph.intor('int1e_ovlp')
    homo_cart = normalize_to_pyscf_basis_fn(
        reorder_mo_coeffs_to_pyscf_fn(fchk, fchk['alpha_mo'][fchk['nalpha'] - 1]),
        S_cart)
    U = build_cart2sph_matrix(mol_cart, mol_sph)
    homo_sph = U @ homo_cart
    norm_sph = float(homo_sph @ S_sph @ homo_sph)
    homo_sph = homo_sph / np.sqrt(norm_sph)  # renormalise (cart->sph projection drops ~0.7%)
    return homo_sph, basis, atom_list, fchk
