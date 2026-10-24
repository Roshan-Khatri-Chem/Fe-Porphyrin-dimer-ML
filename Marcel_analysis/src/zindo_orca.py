"""ZINDO/S via ORCA 6 + fragment-orbital coupling on the dimer.

For each dimer:
  1. Write ORCA input with ZINDO/S keyword + MO printout.
  2. Run ORCA (~10 sec for 74-atom Fe-porphine dimer; longer for Fe-TPP).
  3. Parse ORBITAL ENERGIES + MOLECULAR ORBITALS blocks.
  4. Identify HOMO/HOMO-1 (or the porphine pi pair via projection on monomer
     ZINDO HOMO).
  5. Compute J = (E_HOMO - E_HOMO-1) / 2 for the symmetric homodimer.
"""
import os, re, subprocess, tempfile, shutil
import numpy as np

ORCA = os.path.expanduser('~/WORK/CODE/orca_6_0_0_macosx_openmpi411/orca')
HARTREE_TO_EV = 27.211386245988


def write_zindo_input(atom_list, charge=0, mult=1, path='zindo.inp'):
    inp = '! ZINDO/S\n'
    inp += '%output PrintLevel Normal Print[P_MOs] 1 end\n'
    inp += f'* xyz {charge} {mult}\n'
    for s, p in atom_list:
        inp += f'{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n'
    inp += '*\n'
    with open(path, 'w') as f:
        f.write(inp)


def run_zindo(atom_list, charge=0, mult=1, work_dir=None, cleanup=True):
    """Run ORCA ZINDO/S, return path to .out file."""
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='zindo_')
        own_dir = True
    else:
        os.makedirs(work_dir, exist_ok=True)
        own_dir = False
    inp_path = os.path.join(work_dir, 'zindo.inp')
    out_path = os.path.join(work_dir, 'zindo.out')
    write_zindo_input(atom_list, charge=charge, mult=mult, path=inp_path)
    try:
        subprocess.run([ORCA, 'zindo.inp'], cwd=work_dir,
                       stdout=open(out_path, 'w'), stderr=subprocess.STDOUT,
                       timeout=600)
    except subprocess.TimeoutExpired:
        if own_dir and cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError("ZINDO timeout (>10 min)")
    return out_path, work_dir, own_dir


def parse_zindo_output(out_path):
    """Parse ORCA ZINDO output. Returns dict with eigvals, eigvecs, ao_labels, n_elec."""
    with open(out_path) as f:
        lines = f.readlines()

    # Find ORBITAL ENERGIES block
    e_start = None; mo_start = None; nbasis = None; n_elec = None
    for i, l in enumerate(lines):
        if 'Number of Electrons    NEL' in l:
            n_elec = int(re.search(r'(\d+)', l).group(1))
        elif 'Basis Dimension' in l and 'Dim' in l:
            nbasis = int(re.search(r'(\d+)\s*$', l.rstrip()).group(1))
        elif 'ORBITAL ENERGIES' in l and e_start is None:
            e_start = i
        elif 'MOLECULAR ORBITALS' in l and mo_start is None:
            mo_start = i
    if e_start is None or mo_start is None:
        raise RuntimeError(f"Could not find ORBITAL ENERGIES or MOLECULAR ORBITALS in {out_path}")

    # Parse orbital energies (header has 4 lines, then table until blank line)
    eigvals_eV = []
    j = e_start + 4
    while j < len(lines):
        s = lines[j].strip()
        if not s or 'Only the first' in s or '---' in s:
            break
        toks = s.split()
        if len(toks) >= 4 and toks[0].isdigit():
            eigvals_eV.append(float(toks[3]))
            j += 1
        else:
            break

    # Parse MO coefficients block (orbitals indexed 0 .. nmo_printed-1)
    ao_labels = []
    mo_dict = {}
    i = mo_start + 2
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        toks = s.split()
        if not toks[0].isdigit():
            break
        orb_idx = [int(x) for x in toks]
        i += 1  # energies
        i += 1  # occupations
        i += 1  # ----
        i += 1
        block_labels = []
        block_data = []
        while i < len(lines):
            row = lines[i]
            if not row.strip():
                break
            row_toks = row.split()
            if len(row_toks) < 3:
                break
            try:
                label = ' '.join(row_toks[:2])
                vals = [float(x) for x in row_toks[2:]]
                if len(vals) != len(orb_idx):
                    break
            except ValueError:
                break
            block_labels.append(label)
            block_data.append(vals)
            i += 1
        if not ao_labels:
            ao_labels = block_labels
        block = np.array(block_data)
        for k, oi in enumerate(orb_idx):
            mo_dict[oi] = block[:, k]

    # Construct full MO matrix (nbasis x nmo_printed)
    nmo_printed = max(mo_dict.keys()) + 1
    mocoeffs = np.zeros((nbasis, nmo_printed))
    for oi, c in mo_dict.items():
        if len(c) == nbasis:
            mocoeffs[:, oi] = c
    return {
        'eigvals_eV': np.array(eigvals_eV),
        'mocoeffs': mocoeffs,           # (nbasis, nmo_printed)
        'ao_labels': ao_labels,         # list of length nbasis
        'n_elec': n_elec,
        'nbasis': nbasis,
        'homo_idx': n_elec // 2 - 1,
    }


def parse_atom_label(label):
    """Parse '0C 1pz' -> (atom_idx=0, sym='C', shell='1', l='p', m='z')."""
    m = re.match(r'(\d+)([A-Z][a-z]?)\s*(\d+)([spdf])([a-z0-9]*)', label)
    if not m:
        raise ValueError(f"Cannot parse AO label: {label}")
    return int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5)


def atom_decomp(c_homo, ao_labels):
    """Sum c^2 per atom (assumes orthonormal MOs in minimal basis, like ZINDO)."""
    atom_w = {}
    for lab, c in zip(ao_labels, c_homo):
        atom_idx, sym, _, _, _ = parse_atom_label(lab)
        atom_w[(atom_idx, sym)] = atom_w.get((atom_idx, sym), 0.0) + c ** 2
    return atom_w


def element_decomp(c_homo, ao_labels):
    """Aggregate by element."""
    aw = atom_decomp(c_homo, ao_labels)
    elem = {}
    for (idx, sym), w in aw.items():
        elem[sym] = elem.get(sym, 0.0) + w
    return elem


def J_from_dimer_eigvals(eigvals_eV, homo_idx, n_pi_search=4):
    """For a closed-shell symmetric homodimer, J = (E_HOMO - E_HOMO-1) / 2.
    But the porphine pi pair may not be exactly at HOMO/HOMO-1; search the top
    n_pi_search occupied orbitals for the largest splitting.
    Returns (J_eV, idx_high, idx_low)."""
    # Try splittings between consecutive occupied orbitals near HOMO
    candidates = []
    for di in range(min(n_pi_search, homo_idx)):
        idx_high = homo_idx - di
        idx_low = idx_high - 1
        if idx_low < 0:
            break
        delta = eigvals_eV[idx_high] - eigvals_eV[idx_low]
        candidates.append((delta, idx_high, idx_low))
    # Use HOMO/HOMO-1 by default (largest naive splitting)
    if not candidates:
        return np.nan, -1, -1
    delta, ih, il = max(candidates, key=lambda x: x[0])
    return delta / 2.0, ih, il


def J_from_fragment_projection(eigvals_eV, mocoeffs, ao_labels,
                               c_ref_pi, monomer1_atom_set, monomer2_atom_set):
    """Find dimer MOs with largest projection onto (c_ref_pi at m1, c_ref_pi at m2)
    and use their splitting. Falls back to HOMO/HOMO-1 if projection is ambiguous.

    monomer1_atom_set, monomer2_atom_set : sets of dimer atom indices.
    c_ref_pi : (nbasis_monomer,) ZINDO HOMO coefficients of reference monomer.
               Same atom ordering as ao_labels of the monomer.
    """
    # Identify dimer AO indices belonging to m1 vs m2
    nbasis_dimer, nmo = mocoeffs.shape
    is_m1 = np.zeros(nbasis_dimer, bool)
    is_m2 = np.zeros(nbasis_dimer, bool)
    for i, lab in enumerate(ao_labels):
        atom_idx, sym, _, _, _ = parse_atom_label(lab)
        if atom_idx in monomer1_atom_set:
            is_m1[i] = True
        elif atom_idx in monomer2_atom_set:
            is_m2[i] = True

    # Build "rotated" reference HOMO at m1 and m2 positions: just place the same coefs
    # at the corresponding atoms (assumes atom order in dimer matches reference monomer
    # ordering for each monomer block). This works for atoms-in-same-order dimers.
    # Returns ZINDO equivalent of the c1, c2 vectors.
    nbasis_mono = len(c_ref_pi)
    c1 = np.zeros(nbasis_dimer)
    c2 = np.zeros(nbasis_dimer)
    m1_idx_global = np.where(is_m1)[0]
    m2_idx_global = np.where(is_m2)[0]
    if len(m1_idx_global) != nbasis_mono or len(m2_idx_global) != nbasis_mono:
        raise ValueError(f"AO count mismatch: m1={len(m1_idx_global)}, m2={len(m2_idx_global)}, mono={nbasis_mono}")
    c1[m1_idx_global] = c_ref_pi
    c2[m2_idx_global] = c_ref_pi

    # For ZINDO (orthonormal minimal basis), <c1|MO_j> = c1 . mo_j
    proj1 = np.abs(c1 @ mocoeffs)  # (nmo,) abs projection
    proj2 = np.abs(c2 @ mocoeffs)
    score = proj1 + proj2  # combined projection
    # Find top 2 occupied MOs by score
    n_elec = (np.where(eigvals_eV[:nmo] < 0)[0][-1] + 1) if (eigvals_eV[:nmo] < 0).any() else nmo // 2
    # Just look at HOMO ± window
    # Actually find dimer MOs with largest sum of |proj1| + |proj2|
    top_two = np.argsort(score)[::-1][:2]
    top_two_sorted = sorted(top_two, key=lambda i: eigvals_eV[i], reverse=True)
    ih, il = top_two_sorted[0], top_two_sorted[1]
    delta = eigvals_eV[ih] - eigvals_eV[il]
    return delta / 2.0, ih, il
