"""
Extended Hückel Theory (EHT) on the dimer with fragment-orbital coupling.

H_ii = VOIP for valence orbitals, very negative for cores (effectively decoupled).
H_ij = K * (H_ii + H_jj)/2 * S_ij  (Wolfsberg-Helmholz, K = 1.75)
S_ij = AO overlap from PySCF STO-3G (Slater-fit Gaussians).

Fragment-orbital coupling for the porphine pi HOMO:
  J_eff = (H_12 - 0.5 * (H_11 + H_22) * S_12) / (1 - S_12^2)
where H_ij = c_i^T H_dimer c_j and S_ij = c_i^T S_dimer c_j, with c_i the
reference monomer EHT-HOMO rotated to monomer i's frame via Wigner-D matrices.
"""
import re
import numpy as np
from scipy.linalg import eigh
from pyscf import gto, symm
from scipy.spatial.transform import Rotation

K_WH = 1.75   # Wolfsberg-Helmholz constant
HARTREE_TO_EV = 27.211386245988

# Hoffmann's standard EHT VOIPs (eV) — single-zeta where applicable
EHT_VOIP_EV = {
    ('H',  '1s'): -13.60,
    ('C',  '2s'): -21.40,
    ('C',  '2p'): -11.40,
    ('N',  '2s'): -26.00,
    ('N',  '2p'): -13.40,
    ('O',  '2s'): -32.30,
    ('O',  '2p'): -14.80,
    # Fe: standard Hoffmann/CACAO parameters
    ('Fe', '4s'): -9.17,
    ('Fe', '4p'): -5.32,
    ('Fe', '3d'): -12.60,
}
CORE_VOIP_EV = -1.0e4  # core orbitals: very negative -> decoupled from valence
CORE_VOIP_HA = CORE_VOIP_EV / HARTREE_TO_EV


def _ao_principal_l(label):
    """Parse a PySCF ao_label like '0 Fe 3dxy' or '1 C 2px' into (atom_sym, n, l)."""
    parts = label.split()
    atom_sym = parts[1]
    nl = parts[2]  # e.g. '3dxy', '2s', '2px', '4s'
    m = re.match(r'^(\d+)([spdfg])', nl)
    if not m:
        raise ValueError(f"Cannot parse AO label: {label}")
    n = int(m.group(1))
    l = m.group(2)
    return atom_sym, n, l


def build_eht_hamiltonian(mol, voip_eV=EHT_VOIP_EV, K=K_WH, in_hartree=True):
    """Build EHT Hamiltonian H and overlap S for the given molecule.

    Returns
    -------
    H : (n_ao, n_ao)  Fock matrix in Hartree (or eV if in_hartree=False)
    S : (n_ao, n_ao)  AO overlap (Slater via STO-3G fit)
    """
    S = mol.intor('int1e_ovlp')
    n = mol.nao
    H_diag = np.empty(n)
    for i in range(n):
        sym, npr, l = _ao_principal_l(mol.ao_labels()[i])
        key = (sym, f'{npr}{l}')
        Hii_eV = voip_eV.get(key, CORE_VOIP_EV)
        H_diag[i] = Hii_eV / HARTREE_TO_EV if in_hartree else Hii_eV

    H = np.zeros_like(S)
    for i in range(n):
        H[i, i] = H_diag[i]
    # off-diagonals: vectorized W-H formula
    # H_ij = K * (H_ii + H_jj) / 2 * S_ij
    avg = 0.5 * (H_diag[:, None] + H_diag[None, :])
    off = K * avg * S
    iu = np.triu_indices(n, k=1)
    H[iu] = off[iu]
    H[iu[1], iu[0]] = off[iu]
    return H, S


def n_valence_electrons_neutral(symbols):
    """Total valence electrons assuming neutral atoms (H=1, C=4, N=5, O=6, Fe=8)."""
    table = {'H': 1, 'C': 4, 'N': 5, 'O': 6, 'Fe': 8}  # valence only
    return sum(table[s] for s in symbols)


def homo_idx_from_filling(eigvals, n_elec_total, n_core_per_atom_count=None):
    """For a closed-shell EHT calc, HOMO index = n_elec/2 - 1.
    But because we put core orbitals at very negative VOIPs, the *valence*
    HOMO index is n_elec_valence/2 - 1 + n_core_orbitals."""
    raise NotImplementedError("Use HOMO selection by orbital projection instead")


# -----------------------------------------------------------------------------
# Fragment-orbital coupling
# -----------------------------------------------------------------------------

def kabsch_align(ref, tgt):
    cR = ref.mean(axis=0); cT = tgt.mean(axis=0)
    H = (ref - cR).T @ (tgt - cT)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = cT - R @ cR
    return R, T


def build_ao_rotation_matrix_sph(mol_sph, R):
    """Per-shell Wigner-D rotation matrix for a *spherical* PySCF basis."""
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


def porphine_NFe_align(target_pos, target_sym, ref_pos, ref_sym):
    """Kabsch-align the ref Fe + 4 porphyrin core N onto the target's Fe + core N.
    Tries 8 D4 permutations of the N labelling; picks lowest RMSD."""
    ref_pos = np.asarray(ref_pos); ref_sym = np.asarray(ref_sym)
    tgt_pos = np.asarray(target_pos); tgt_sym = np.asarray(target_sym)

    ref_fe = ref_pos[ref_sym == 'Fe'][0]
    ref_N_all = ref_pos[ref_sym == 'N']
    d_ref = np.linalg.norm(ref_N_all - ref_fe, axis=1)
    ref_N = ref_N_all[np.argsort(d_ref)[:4]]

    tgt_fe = tgt_pos[tgt_sym == 'Fe'][0]
    tgt_N_all = tgt_pos[tgt_sym == 'N']
    d_tgt = np.linalg.norm(tgt_N_all - tgt_fe, axis=1)
    tgt_N = tgt_N_all[np.argsort(d_tgt)[:4]]

    def sort_by_angle(N_pos, fe_pos):
        c = N_pos - fe_pos
        _, _, Vt = np.linalg.svd(c)
        normal = Vt[-1]
        x_pre = c[0] - np.dot(c[0], normal) * normal
        x_pre /= np.linalg.norm(x_pre)
        y_pre = np.cross(normal, x_pre)
        ang = np.arctan2(c @ y_pre, c @ x_pre)
        return N_pos[np.argsort(ang)]

    ref_N_s = sort_by_angle(ref_N, ref_fe)
    tgt_N_s = sort_by_angle(tgt_N, tgt_fe)

    best_R, best_T, best_rmsd = None, None, np.inf
    for k in range(4):
        for refl in [False, True]:
            perm = np.array([(k - i) % 4 if refl else (k + i) % 4 for i in range(4)])
            ref_5 = np.vstack([[ref_fe], ref_N_s])
            tgt_5 = np.vstack([[tgt_fe], tgt_N_s[perm]])
            R, T = kabsch_align(ref_5, tgt_5)
            rmsd = np.sqrt(np.mean(np.sum((ref_5 @ R.T + T - tgt_5) ** 2, axis=1)))
            if rmsd < best_rmsd:
                best_rmsd, best_R, best_T = rmsd, R, T
    return best_R, best_T


class EHTCoupling:
    """EHT-based fragment-orbital electronic coupling for a homodimer.

    Reference monomer is run once; for each dimer, the reference monomer is
    Kabsch-aligned to each monomer (porphine core only — robust to TPP / peptoid
    substituents), the AO basis is placed at the idealised positions, and the
    reference HOMO is rotated via Wigner-D matrices to obtain the monomer-frame
    fragment orbital in the dimer basis."""

    def __init__(self, ref_atom_list, basis='sto-3g', target_a2u=True,
                 a2u_search_window=8):
        """
        target_a2u : if True, identify the porphine a2u-like reference orbital
                     by node-pattern score (large amplitude on alpha-C + meso-C,
                     small on N + beta-C) within +/- a2u_search_window of HOMO.
                     If False, use the strict HOMO from valence filling.
        """
        self.basis = basis
        self.ref_atoms = ref_atom_list
        self.ref_syms = [s for s, _ in ref_atom_list]
        self.ref_coords = np.array([p for _, p in ref_atom_list])

        self.ref_mol = gto.M(atom=ref_atom_list, basis=basis,
                             unit='Angstrom', verbose=0, cart=False)
        H_ref, S_ref = build_eht_hamiltonian(self.ref_mol)
        eigvals, eigvecs = eigh(H_ref, S_ref)

        # Strict HOMO from valence filling
        n_elec = n_valence_electrons_neutral(self.ref_syms)
        n_core = sum(self._n_core_orbitals(s) for s in self.ref_syms)
        n_doubly_occ = n_elec // 2
        homo_idx_strict = n_core + n_doubly_occ - 1

        # Identify atom roles for a2u scoring
        ref_pos = self.ref_coords
        syms = np.asarray(self.ref_syms)
        fe_idx = int(np.where(syms == 'Fe')[0][0])
        N_all = np.where(syms == 'N')[0]
        d_to_fe = np.linalg.norm(ref_pos[N_all] - ref_pos[fe_idx], axis=1)
        core_N = N_all[np.argsort(d_to_fe)[:4]]
        C_all = np.where(syms == 'C')[0]
        alpha_C = []
        for ci in C_all:
            d_N = np.linalg.norm(ref_pos[ci] - ref_pos[core_N], axis=1)
            if np.min(d_N) < 1.6:
                alpha_C.append(ci)
        beta_C, meso_C = [], []
        for ci in C_all:
            if ci in alpha_C: continue
            d_a = np.linalg.norm(ref_pos[ci] - ref_pos[alpha_C], axis=1)
            bonded = np.array(alpha_C)[d_a < 1.6]
            if len(bonded) >= 2:
                meso_C.append(ci)
            elif len(bonded) == 1:
                beta_C.append(ci)
        self._alpha_C = alpha_C
        self._beta_C = beta_C
        self._meso_C = meso_C
        self._core_N = core_N
        self._fe_idx = fe_idx

        slices = self.ref_mol.aoslice_by_atom()
        def atom_w(c_, idx):
            s_, e_ = slices[idx, 2], slices[idx, 3]
            return float(c_[s_:e_] @ (S_ref @ c_)[s_:e_])
        def a2u_score(c_):
            wa = sum(atom_w(c_, i) for i in alpha_C)
            wm = sum(atom_w(c_, i) for i in meso_C)
            wb = sum(atom_w(c_, i) for i in beta_C)
            wn = sum(atom_w(c_, i) for i in core_N)
            return wa + wm - wb - abs(wn)

        if target_a2u:
            best_idx, best_score = homo_idx_strict, -np.inf
            for di in range(-a2u_search_window, a2u_search_window + 1):
                idx = homo_idx_strict + di
                if 0 <= idx < len(eigvals):
                    sc = a2u_score(eigvecs[:, idx])
                    if sc > best_score:
                        best_score, best_idx = sc, idx
            homo_idx_global = best_idx
        else:
            homo_idx_global = homo_idx_strict

        c_homo = eigvecs[:, homo_idx_global]
        cn_frac, fe_frac, h_frac = self._mulliken_atom_decomp(c_homo, S_ref)
        self.homo_idx = homo_idx_global
        self.homo_idx_strict = homo_idx_strict
        self.homo_E_eV = eigvals[homo_idx_global] * HARTREE_TO_EV
        self.homo_C_N_frac = cn_frac
        self.homo_Fe_frac = fe_frac
        self.c_ref_homo = c_homo
        self.S_ref = S_ref
        self.H_ref_diag = eigvals
        self.ref_eigvals = eigvals
        self.ref_eigvecs = eigvecs

    def _n_core_orbitals(self, sym):
        """Number of core (1s, 2s, 2p ...) orbitals to skip for valence filling."""
        # Closed-shell core: H=0; C/N/O = 1 (1s); Fe = 5 (1s,2s,2p) + ... let's count properly
        cores = {'H': 0, 'C': 1, 'N': 1, 'O': 1, 'Fe': 9}
        # Fe: 1s + 2s + 2px,2py,2pz + 3s + 3px,3py,3pz = 1 + 1 + 3 + 1 + 3 = 9
        return cores[sym]

    def _mulliken_atom_decomp(self, c, S):
        """Compute (CN_frac, Fe_frac, H_frac) from c, S."""
        Sc = S @ c
        atom_w = {sym: 0.0 for sym in set(self.ref_syms)}
        slices = self.ref_mol.aoslice_by_atom()
        for atom_idx in range(self.ref_mol.natm):
            sym = self.ref_syms[atom_idx]
            s_, e_ = slices[atom_idx, 2], slices[atom_idx, 3]
            atom_w[sym] = atom_w.get(sym, 0.0) + float(c[s_:e_] @ Sc[s_:e_])
        cn = atom_w.get('C', 0) + atom_w.get('N', 0)
        fe = atom_w.get('Fe', 0)
        h  = atom_w.get('H', 0)
        tot = cn + fe + h
        return cn/tot, fe/tot, h/tot

    def compute(self, m1_pos, m1_sym, m2_pos, m2_sym):
        """Compute EHT fragment-orbital coupling J (eV) for one dimer."""
        # 1. Kabsch-align reference porphine core onto each monomer
        R1, T1 = porphine_NFe_align(m1_pos, m1_sym, self.ref_coords, self.ref_syms)
        R2, T2 = porphine_NFe_align(m2_pos, m2_sym, self.ref_coords, self.ref_syms)

        # 2. Build dimer with reference monomer atoms at idealised positions
        ref_at_m1 = (self.ref_coords @ R1.T + T1).tolist()
        ref_at_m2 = (self.ref_coords @ R2.T + T2).tolist()
        atoms = [(s, p) for s, p in zip(self.ref_syms, ref_at_m1)] + \
                [(s, p) for s, p in zip(self.ref_syms, ref_at_m2)]
        mol_d = gto.M(atom=atoms, basis=self.basis, unit='Angstrom',
                      verbose=0, cart=False)

        # 3. EHT H, S for dimer
        H_d, S_d = build_eht_hamiltonian(mol_d)

        # 4. Rotate reference HOMO to each monomer's frame via Wigner-D
        M1 = build_ao_rotation_matrix_sph(self.ref_mol, R1)
        M2 = build_ao_rotation_matrix_sph(self.ref_mol, R2)
        c1_rot = M1 @ self.c_ref_homo
        c2_rot = M2 @ self.c_ref_homo

        nao = self.ref_mol.nao
        c1 = np.zeros(mol_d.nao)
        c2 = np.zeros(mol_d.nao)
        c1[:nao] = c1_rot
        c2[nao:nao + nao] = c2_rot

        # 5. Fragment-orbital coupling (Lowdin orthogonalised effective)
        H_11 = float(c1 @ H_d @ c1)
        H_22 = float(c2 @ H_d @ c2)
        H_12 = float(c1 @ H_d @ c2)
        S_12 = float(c1 @ S_d @ c2)
        denom = (1.0 - S_12 ** 2)
        if abs(denom) < 1e-12:
            return np.nan, S_12, H_12
        J_ha = (H_12 - 0.5 * (H_11 + H_22) * S_12) / denom
        J_eV = J_ha * HARTREE_TO_EV
        return J_eV, S_12, H_12 * HARTREE_TO_EV
