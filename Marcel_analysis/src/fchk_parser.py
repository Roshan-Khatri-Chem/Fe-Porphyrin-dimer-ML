"""Minimal Q-Chem/Gaussian fchk parser for the SRSH-PCM monomer files.

Extracts:
  - geometry (Bohr -> Angstrom)
  - basis set as a PySCF-compatible dict (per element)
  - alpha/beta MO coefficients in the order PySCF expects (after a
    per-Cartesian-d normalisation rescaling, see notes below)

Notes on conventions:
  * Q-Chem 6-31G(d) uses Cartesian d (6 functions per shell, shell-type code
    +2). The fchk also stores Cartesian d on Fe (2 sets) and one spherical f
    set (-3) on Fe.
  * The HOMO/SOMO of Fe-porphine in the SRSH-PCM calculation is 100% C+N
    pi-system — the Fe AOs have ZERO coefficient — so the f shell on Fe
    contributes nothing to the relevant orbital.
  * Q-Chem Cartesian d contraction coefficients are normalised so that the
    *sum* over the 6 components equals 1, with d_xx, d_yy, d_zz carrying a
    factor 1/sqrt(3) less than d_xy, d_xz, d_yz (Gaussian convention). PySCF
    Cartesian d normalises each component to 1, so we rescale the dxx, dyy,
    dzz coefficients by sqrt(3) when remapping. The standard test is c.S.c
    == 1 for the HOMO; if that holds we know the convention is consistent.
"""
import numpy as np
import re
import os

# fchk shell type codes
SHELL_TYPES = {
    0:  ('s', 1),   # 1 s function
    -1: ('sp', 4),  # 1 s + 3 p combined
    1:  ('p', 3),
    2:  ('d_cart', 6),
    -2: ('d_sph', 5),
    3:  ('f_cart', 10),
    -3: ('f_sph', 7),
}

# AO ordering within each shell (Q-Chem / Gaussian fchk convention)
# The fchk order matches Gaussian, which writes Cartesian d as:
#   d_xx, d_yy, d_zz, d_xy, d_xz, d_yz
# PySCF Cartesian d order (cart=True) is:
#   d_xx, d_xy, d_xz, d_yy, d_yz, d_zz   (lexicographic)
# So we need a permutation 0,3,4,1,5,2  (fchk -> pyscf) when copying coefficients.
QCHEM_TO_PYSCF_D_CART = [0, 3, 4, 1, 5, 2]
# Gaussian Cartesian f: xxx,yyy,zzz,xyy,xxy,xxz,xzz,yzz,yyz,xyz
# PySCF Cartesian f:    xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz
QCHEM_TO_PYSCF_F_CART = [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]
# Spherical f (-3): Gaussian/Q-Chem ordering is m = 0,+1,-1,+2,-2,+3,-3
# PySCF ordering for spherical f (default) is m = -3,-2,-1,0,+1,+2,+3
QCHEM_TO_PYSCF_F_SPH = [6, 4, 2, 0, 1, 3, 5]
# Spherical d (-2) similarly m = 0,+1,-1,+2,-2 -> PySCF -2,-1,0,+1,+2
QCHEM_TO_PYSCF_D_SPH = [4, 2, 0, 1, 3]

PERIODIC = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 26: 'Fe'}
BOHR_TO_A = 0.5291772083


def _read_block(lines, start_index, n_values):
    """Read n_values floats/ints starting after `start_index` (0-based, header line included)."""
    vals = []
    i = start_index + 1
    while len(vals) < n_values:
        for tok in lines[i].split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        i += 1
    return vals[:n_values], i


def parse_fchk(path):
    with open(path) as f:
        lines = f.readlines()

    sections = {}
    for i, line in enumerate(lines):
        # parse declaration lines like "Name  I  N=  count"  or scalar "Name  I    value"
        m_arr = re.match(r'^(.{40})\s+([IRC])\s+N=\s*(\d+)\s*$', line)
        m_sca = re.match(r'^(.{40})\s+([IRC])\s+([-\d\.E+]+)\s*$', line)
        if m_arr:
            name = m_arr.group(1).strip()
            kind = m_arr.group(2)
            n = int(m_arr.group(3))
            sections[name] = ('array', kind, n, i)
        elif m_sca:
            name = m_sca.group(1).strip()
            kind = m_sca.group(2)
            val = m_sca.group(3)
            sections[name] = ('scalar', kind, val, i)

    def get_array(name):
        kind = sections[name][1]
        n = sections[name][2]
        i = sections[name][3]
        vals, _ = _read_block(lines, i, n)
        if kind == 'I':
            return np.array(vals, dtype=int)
        return np.array(vals, dtype=float)

    natom = int(sections['Number of atoms'][2])
    nbasis = int(sections['Number of basis functions'][2])
    charge = int(sections['Charge'][2])
    mult = int(sections['Multiplicity'][2])

    atomnos = get_array('Atomic numbers').astype(int)
    coords_bohr = get_array('Current cartesian coordinates').reshape(natom, 3)
    coords_ang = coords_bohr * BOHR_TO_A

    shell_types = get_array('Shell types').astype(int)
    nprim_per_shell = get_array('Number of primitives per shell').astype(int)
    shell_to_atom = get_array('Shell to atom map').astype(int) - 1  # 0-based
    prim_exps = get_array('Primitive exponents')
    prim_coefs = get_array('Contraction coefficients')
    sp_coefs = get_array('P(S=P) Contraction coefficients') if 'P(S=P) Contraction coefficients' in sections else np.zeros_like(prim_coefs)

    alpha_mo = get_array('Alpha MO coefficients').reshape(nbasis, nbasis)
    beta_mo = get_array('Beta MO coefficients').reshape(nbasis, nbasis)
    alpha_e = get_array('Alpha Orbital Energies')
    beta_e = get_array('Beta Orbital Energies')

    nelec = int(sections['Number of electrons'][2])
    nalpha = int(sections['Number of alpha electrons'][2])
    nbeta = int(sections['Number of beta electrons'][2])

    return dict(
        natom=natom, nbasis=nbasis, charge=charge, mult=mult,
        nalpha=nalpha, nbeta=nbeta,
        atomnos=atomnos, coords_ang=coords_ang,
        shell_types=shell_types, nprim_per_shell=nprim_per_shell,
        shell_to_atom=shell_to_atom, prim_exps=prim_exps,
        prim_coefs=prim_coefs, sp_coefs=sp_coefs,
        alpha_mo=alpha_mo, beta_mo=beta_mo,
        alpha_e=alpha_e, beta_e=beta_e,
    )


def build_pyscf_basis(fchk):
    """Build a PySCF basis dict from the fchk shell info (per-element)."""
    basis = {}
    prim_offset = 0
    # Group shells by atom, then by element (use first atom of each element seen)
    atom_shells = [[] for _ in range(fchk['natom'])]
    for shell_idx, n_prim in enumerate(fchk['nprim_per_shell']):
        st = fchk['shell_types'][shell_idx]
        atom = fchk['shell_to_atom'][shell_idx]
        exps = fchk['prim_exps'][prim_offset:prim_offset + n_prim]
        coefs = fchk['prim_coefs'][prim_offset:prim_offset + n_prim]
        sp_c = fchk['sp_coefs'][prim_offset:prim_offset + n_prim]
        atom_shells[atom].append((st, exps.copy(), coefs.copy(), sp_c.copy()))
        prim_offset += n_prim

    seen_elements = set()
    for atom_idx in range(fchk['natom']):
        sym = PERIODIC[fchk['atomnos'][atom_idx]]
        if sym in seen_elements:
            continue
        seen_elements.add(sym)
        shells = []
        for st, exps, coefs, sp_c in atom_shells[atom_idx]:
            if st == 0:    # s
                shells.append([0] + [[float(e), float(c)] for e, c in zip(exps, coefs)])
            elif st == 1:  # p
                shells.append([1] + [[float(e), float(c)] for e, c in zip(exps, coefs)])
            elif st == -1:  # sp combined
                shells.append([0] + [[float(e), float(c)] for e, c in zip(exps, coefs)])
                shells.append([1] + [[float(e), float(c)] for e, c in zip(exps, sp_c)])
            elif st == 2 or st == -2:  # d (Cartesian or spherical — pyscf will use mol.cart)
                shells.append([2] + [[float(e), float(c)] for e, c in zip(exps, coefs)])
            elif st == 3 or st == -3:  # f
                shells.append([3] + [[float(e), float(c)] for e, c in zip(exps, coefs)])
            else:
                raise ValueError(f'Unsupported shell type {st}')
        basis[sym] = shells
    return basis, atom_shells


def pyscf_shell_size(st):
    """AO count for a shell in PySCF when mol.cart=True (Cartesian everywhere)."""
    if st == 0: return 1
    if st in (1, -1):  # SP -> p (s handled separately) or just p
        return 3 if st == 1 else 4  # only used for sp_size accounting; sp split below
    if st in (2, -2): return 6   # cart d
    if st in (3, -3): return 10  # cart f
    raise ValueError(st)


def _shell_l(st):
    """Angular momentum for a fchk shell type (sp -> ('s','p') handled separately)."""
    if st == 0: return 0
    if st in (1, -1): return 1  # p (sp split into s+p; the p part)
    if st in (2, -2): return 2
    if st in (3, -3): return 3
    raise ValueError(st)


def reorder_mo_coeffs_to_pyscf(fchk, mo_vec):
    """
    Take an MO coefficient vector in fchk AO ordering and reorder it to the
    PySCF basis ordering for cart=True.

    Two reorderings happen:
      1. fchk AO ordering inside a shell -> pyscf ordering (Cart d/f permutations)
      2. fchk shell ordering inside an atom -> pyscf shell ordering: PySCF
         groups all shells of the same angular momentum together (l=0 first,
         then l=1, then l=2, ...) regardless of input order.

    Spherical d/f shells (-2, -3) are padded with zeros for the extra
    Cartesian components.  This is exact only when the MO has zero coefficient
    on that shell — verify by checking c^T S c == 1.
    """
    # Group shells by atom
    shells_per_atom = {}
    in_off = 0
    for shell_idx, st in enumerate(fchk['shell_types']):
        atom = fchk['shell_to_atom'][shell_idx]
        n_in = SHELL_TYPES[st][1]
        # for sp shells (-1) the fchk packs [s, px, py, pz] together
        coefs_block = mo_vec[in_off:in_off + n_in]
        shells_per_atom.setdefault(atom, []).append((st, coefs_block))
        in_off += n_in
    assert in_off == len(mo_vec)

    nao_pyscf = sum(pyscf_shell_size_full(st) for st in fchk['shell_types'])
    out = np.zeros(nao_pyscf)
    out_off = 0

    for atom_idx in range(fchk['natom']):
        # Build a list of (l, sub_block, fchk_st) per emitted PySCF shell.
        # SP shells get split into (l=0, s_part) + (l=1, p_part).
        emitted = []
        for st, block in shells_per_atom[atom_idx]:
            if st == -1:
                emitted.append((0, block[0:1], 0))    # s part
                emitted.append((1, block[1:4], 1))    # p part
            else:
                emitted.append((_shell_l(st), block, st))
        # Sort by l (stable to preserve input order within same l)
        emitted.sort(key=lambda x: x[0])

        for l, block, st_orig in emitted:
            if l == 0:
                out[out_off:out_off+1] = block
                out_off += 1
            elif l == 1:
                out[out_off:out_off+3] = block
                out_off += 3
            elif l == 2:
                if st_orig == 2:  # Cart d
                    for k_pyscf, k_fchk in enumerate(QCHEM_TO_PYSCF_D_CART):
                        out[out_off + k_pyscf] = block[k_fchk]
                    out_off += 6
                else:  # spherical d -> pad to 6 Cart with zeros
                    if np.max(np.abs(block)) > 1e-8:
                        raise NotImplementedError('Sph d -> Cart d with non-zero coeffs not implemented')
                    out_off += 6
            elif l == 3:
                if st_orig == 3:  # Cart f
                    for k_pyscf, k_fchk in enumerate(QCHEM_TO_PYSCF_F_CART):
                        out[out_off + k_pyscf] = block[k_fchk]
                    out_off += 10
                else:  # spherical f -> pad to 10 Cart with zeros (exact only if block ~ 0)
                    out_off += 10
            else:
                raise ValueError(f'Unsupported l={l}')
    return out


def pyscf_shell_size_full(st):
    """AO count for a shell in PySCF when mol.cart=True (used by reorder)."""
    if st == 0: return 1
    if st == 1: return 3
    if st == -1: return 4    # sp split into s+p, but total is still 4
    if st in (2, -2): return 6
    if st in (3, -3): return 10
    raise ValueError(st)


def normalize_to_pyscf_basis(c_pyscf_perm, S_pyscf):
    """Convert MO coeffs from a per-AO-normalised basis (e.g. Q-Chem fchk) to
    the un-normalised PySCF Cartesian basis by dividing by sqrt(diag(S_pyscf)).

    After this rescaling, c^T S_pyscf c equals the original c_fchk^T S_fchk c_fchk
    (which is 1 for normalised orbitals)."""
    return c_pyscf_perm / np.sqrt(S_pyscf.diagonal())


def get_atombasis_pyscf_order(fchk):
    """Return list of AO index slices per atom in PySCF (cart=True) basis order."""
    sizes_per_atom = [0] * fchk['natom']
    for shell_idx, st in enumerate(fchk['shell_types']):
        atom = fchk['shell_to_atom'][shell_idx]
        sizes_per_atom[atom] += pyscf_shell_size_full(st)
    bounds = []
    off = 0
    for n in sizes_per_atom:
        bounds.append((off, off + n))
        off += n
    return bounds


if __name__ == '__main__':
    import sys
    from pyscf import gto

    fchk_path = '/Users/mbaer/Projects/ML_for_coupling/data/from_slater/additional/fe_mon_fchk/fe2_sp.in.fchk'
    if len(sys.argv) > 1:
        fchk_path = sys.argv[1]

    fchk = parse_fchk(fchk_path)
    basis, _ = build_pyscf_basis(fchk)

    syms = [PERIODIC[z] for z in fchk['atomnos']]
    atom_list = [(s, p.tolist()) for s, p in zip(syms, fchk['coords_ang'])]
    mol = gto.M(atom=atom_list, basis=basis, unit='Angstrom',
                charge=fchk['charge'], spin=fchk['mult']-1,
                verbose=0, cart=True)
    expected_pyscf_nao = sum(pyscf_shell_size_full(st) for st in fchk['shell_types'])
    print(f'mol.nao = {mol.nao}, fchk.nbasis = {fchk["nbasis"]}, '
          f'expected_pyscf_nao = {expected_pyscf_nao}')
    if mol.nao != expected_pyscf_nao:
        print('Basis size mismatch — check d/f shell handling')
        sys.exit(1)

    # Test HOMO normalisation
    S = mol.intor('int1e_ovlp')
    homo_idx = fchk['nalpha'] - 1
    c_a = fchk['alpha_mo'][homo_idx]  # row indexed by MO; cols = AO
    c_a_pyscf = normalize_to_pyscf_basis(reorder_mo_coeffs_to_pyscf(fchk, c_a), S)
    norm_a = float(c_a_pyscf @ S @ c_a_pyscf)
    print(f'Alpha HOMO (idx={homo_idx}, E={fchk["alpha_e"][homo_idx]:.3f}eV) c.S.c = {norm_a:.4f} (expect 1.0)')

    if fchk['nbeta'] > 0:
        homo_b = fchk['nbeta'] - 1
        c_b = fchk['beta_mo'][homo_b]
        c_b_pyscf = normalize_to_pyscf_basis(reorder_mo_coeffs_to_pyscf(fchk, c_b), S)
        norm_b = float(c_b_pyscf @ S @ c_b_pyscf)
        print(f'Beta  HOMO (idx={homo_b}, E={fchk["beta_e"][homo_b]:.3f}eV) c.S.c = {norm_b:.4f}')

    # HOMO atom decomposition
    bounds = get_atombasis_pyscf_order(fchk)
    fe_w = cn_w = h_w = 0.0
    for atom_idx, (s_, e_) in enumerate(bounds):
        sym = PERIODIC[fchk['atomnos'][atom_idx]]
        # Mulliken-ish via c.S.c restricted to atom block
        c_block = c_a_pyscf[s_:e_]
        S_block = S[s_:e_, :]
        contrib = float(c_block @ S_block @ c_a_pyscf)
        if sym == 'Fe': fe_w += contrib
        elif sym in ('C', 'N'): cn_w += contrib
        elif sym == 'H': h_w += contrib
    tot = fe_w + cn_w + h_w
    print(f'Alpha HOMO Mulliken: Fe={fe_w/tot*100:.1f}%, C+N={cn_w/tot*100:.1f}%, H={h_w/tot*100:.1f}%  (sum={tot:.4f})')
