"""DFTB+ runner + parser for fragment-orbital electronic coupling.

Workflow (mirrors zindo_orca.py):
  1. Write dftb_in.hsd with the dimer geometry, mio + trans3d SK files.
  2. Run dftb+ via the conda env binary.
  3. Parse band.out (eigenvalues, occupations) and eigenvec.out (MO coefficients
     with Mulliken populations per AO row).
  4. Identify the porphine pi pair by character (>=95% C+N, ~50/50 m1/m2 split).
  5. Compute J via the Lowdin diabatic formula.
"""
import os, re, subprocess, tempfile, shutil
import numpy as np

DFTB = os.path.expanduser('~/miniconda3/envs/dftb/bin/dftb+')
SK_DIR = os.path.expanduser('~/WORK/MLIP/DFTB/slako_combined/')


def write_dftb_input(atom_list, work_dir, charge=0.0, fermi_K=1.0):
    """Write dftb_in.hsd. atom_list = [(sym, [x,y,z]), ...]."""
    elem_order = ['H', 'C', 'N', 'Fe']
    type_idx = {e: i + 1 for i, e in enumerate(elem_order)}
    sorted_atoms = sorted(atom_list, key=lambda x: elem_order.index(x[0]))
    n = len(sorted_atoms)
    lines = [f'Geometry = GenFormat {{', f'{n}  C', ' '.join(elem_order)]
    for i, (sym, p) in enumerate(sorted_atoms, 1):
        lines.append(f'  {i} {type_idx[sym]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}')
    lines.append('}')
    body = '\n'.join(lines) + f'''

Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = 1e-5
  MaxSCCIterations = 200
  MaxAngularMomentum {{
    H = "s"
    C = "p"
    N = "p"
    Fe = "d"
  }}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{SK_DIR}"
    Separator = "-"
    Suffix = ".skf"
  }}
  Filling = Fermi {{ Temperature [K] = {fermi_K} }}
  Charge = {charge}
  Mixer = Broyden {{ MixingParameter = 0.05 }}
}}

Options {{
  WriteDetailedOut = Yes
}}

Analysis {{
  WriteEigenvectors = Yes
  EigenvectorsAsText = Yes
  WriteBandOut = Yes
  MullikenAnalysis = Yes
}}

ParserOptions {{ ParserVersion = 14 }}
'''
    with open(os.path.join(work_dir, 'dftb_in.hsd'), 'w') as f:
        f.write(body)


def run_dftb(atom_list, charge=0.0, fermi_K=1.0, work_dir=None, cleanup=True):
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='dftb_')
        own = True
    else:
        os.makedirs(work_dir, exist_ok=True)
        own = False
    write_dftb_input(atom_list, work_dir, charge=charge, fermi_K=fermi_K)
    log = open(os.path.join(work_dir, 'dftb.log'), 'w')
    try:
        subprocess.run([DFTB], cwd=work_dir, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    except subprocess.TimeoutExpired:
        log.close()
        if own and cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"DFTB+ timeout (>10 min) in {work_dir}")
    log.close()
    return work_dir, own


def parse_dftb_outputs(work_dir):
    """Parse band.out and eigenvec.out from a finished dftb+ run.
    Returns dict with: eigvals_eV (n_mo,), occupations (n_mo,),
                       pop_per_atom (n_mo, n_atom) [Mulliken],
                       atom_syms (n_atom,), atom_indices_per_mo
    """
    band_path = os.path.join(work_dir, 'band.out')
    eig_path = os.path.join(work_dir, 'eigenvec.out')
    if not (os.path.exists(band_path) and os.path.exists(eig_path)):
        raise RuntimeError(f"DFTB+ outputs missing in {work_dir}")

    eigs = []
    occs = []
    with open(band_path) as f:
        for line in f:
            toks = line.split()
            if len(toks) >= 3:
                try:
                    int(toks[0]); eigs.append(float(toks[1])); occs.append(float(toks[2]))
                except ValueError:
                    pass
    eigvals = np.array(eigs)
    occupations = np.array(occs)

    # Parse eigenvec.out: blocks "Eigenvector: N    (up)"
    with open(eig_path) as f:
        text = f.read()
    blocks = re.split(r'Eigenvector:\s*(\d+)\s+\(up\)', text)
    if len(blocks) < 3:
        raise RuntimeError(f"Could not parse eigenvec.out in {work_dir}")
    n_mo = (len(blocks) - 1) // 2

    # Discover atom info from first block
    atom_sym_per_atom = {}
    n_ao_per_atom = {}
    for line in blocks[2].split('\n'):
        m = re.match(r'\s*(\d+)\s+(\w+)\s+', line)
        if m:
            ai = int(m.group(1)); sym = m.group(2)
            atom_sym_per_atom[ai] = sym
    # (more robust: build from full file)
    n_atoms = max(atom_sym_per_atom.keys())
    atom_syms = [atom_sym_per_atom[i] for i in range(1, n_atoms + 1)]

    # Parse Mulliken populations per atom for each MO
    pop_per_atom = np.zeros((n_mo, n_atoms))
    for i in range(1, len(blocks) - 1, 2):
        idx = int(blocks[i]) - 1  # 0-based
        if idx >= n_mo:
            continue
        content = blocks[i + 1]
        current_atom = None
        for line in content.split('\n'):
            m = re.match(r'\s*(\d+)\s+(\w+)\s+(\w+)\s+([-\d.E]+)\s+([-\d.E]+)', line)
            if m:
                current_atom = int(m.group(1))
                pop = float(m.group(5))
                pop_per_atom[idx, current_atom - 1] += pop
            else:
                m2 = re.match(r'\s+(\w+)\s+([-\d.E]+)\s+([-\d.E]+)', line)
                if m2 and current_atom is not None:
                    pop_per_atom[idx, current_atom - 1] += float(m2.group(3))

    return {
        'eigvals_eV': eigvals,
        'occupations': occupations,
        'pop_per_atom': pop_per_atom,  # (n_mo, n_atom)
        'atom_syms': atom_syms,
    }


def find_pi_pair_J_lowdin(parsed, n_atoms_per_mono, cn_thresh=0.90,
                           search_window=12, atom_offset=0):
    """Like the ZINDO version but using Mulliken populations from DFTB+ output."""
    eigs = parsed['eigvals_eV']
    occs = parsed['occupations']
    pop = parsed['pop_per_atom']
    syms = parsed['atom_syms']
    n_atoms = len(syms)

    # Indices (in the DFTB-internally-reordered atom list) of monomer 1 and 2
    # The reordering by element groups H first, then C, then N, then Fe.
    # We rely on the caller to handle this properly. Here we just use the
    # provided atom_offset and n_atoms_per_mono assuming the dimer atoms are
    # presented with monomer 1 first then monomer 2 (after reordering, this
    # only works if the reordering interleaves the two monomers' atoms by
    # element). Workaround: compute m1/m2 weights from the *element-grouped*
    # atom list; we instead pass a per-atom monomer assignment.
    raise NotImplementedError("Use find_pi_pair_J_lowdin_with_assignment instead.")


def element_decomp(pop_row, syms):
    """Sum populations per element."""
    out = {}
    for s, p in zip(syms, pop_row):
        out[s] = out.get(s, 0.0) + p
    return out


def find_pi_pair_J(parsed, m1_atom_indices, m2_atom_indices,
                   cn_thresh=0.85, search_window=30, allow_nonadjacent=True):
    """Find the highest occupied pi pair (>=cn_thresh C+N character) and
    compute J via Lowdin diabatic formula.

    m1_atom_indices, m2_atom_indices : 0-based indices into the (reordered)
        atom list of the parsed DFTB+ output.

    With allow_nonadjacent=True we look for the two highest-energy pi
    candidates whose energies are close (|dE| < 1.0 eV) even if Fe-d
    orbitals sit between them in the dimer spectrum. The Lowdin formula
    still works as long as the two candidates are bonding/antibonding
    combinations of the same monomer pi.

    Returns (J_eV, idx_high, idx_low, candidates).
    """
    eigs = parsed['eigvals_eV']
    occs = parsed['occupations']
    pop = parsed['pop_per_atom']  # (n_mo, n_atom)
    syms = parsed['atom_syms']
    n_mo = len(eigs)

    # "HOMO" = last orbital with occupation > 0.5
    homo = int(np.where(occs > 0.5)[0][-1]) if (occs > 0.5).any() else n_mo // 2 - 1

    is_cn = np.array([s in ('C', 'N') for s in syms])

    candidates = []
    for i in range(homo + 2, max(homo - search_window, -1), -1):
        if i < 0 or i >= n_mo: continue
        tot = pop[i].sum()
        if tot < 0.1: continue
        cn = pop[i, is_cn].sum() / tot
        if cn < cn_thresh: continue
        w1 = pop[i, m1_atom_indices].sum() / tot
        w2 = pop[i, m2_atom_indices].sum() / tot
        s = w1 + w2
        if s < 0.5: continue
        candidates.append({'idx': i, 'E': eigs[i], 'cn': cn,
                           'w1': w1 / s, 'w2': w2 / s, 'occ': occs[i]})

    if len(candidates) < 2:
        return np.nan, -1, -1, candidates

    # Sort by energy (descending)
    candidates.sort(key=lambda c: -c['E'])
    if allow_nonadjacent:
        # Try all pairs with |dE| < 1 eV; pick the highest-energy valid pair
        for j in range(len(candidates) - 1):
            for k in range(j + 1, len(candidates)):
                ch = candidates[j]
                cl = candidates[k]
                dE = abs(ch['E'] - cl['E'])
                if dE > 1.0: continue
                eps1 = ch['w1'] * ch['E'] + cl['w1'] * cl['E']
                eps2 = ch['w2'] * ch['E'] + cl['w2'] * cl['E']
                delta = (ch['E'] - cl['E']) / 2
                site = (eps1 - eps2) / 2
                if abs(delta) <= abs(site): continue
                # Quality check: the pair should be bonding/antibonding,
                # i.e., w1 of one ~ w2 of other (orthogonal localization)
                # OR both are ~50/50 (delocalized)
                J = float(np.sqrt(delta ** 2 - site ** 2))
                return J, ch['idx'], cl['idx'], candidates[:5]
    else:
        # Original strict adjacency check
        candidates.sort(key=lambda c: -c['idx'])
        for j in range(len(candidates) - 1):
            ch = candidates[j]; cl = candidates[j + 1]
            if abs(ch['idx'] - cl['idx']) > 2: continue
            eps1 = ch['w1'] * ch['E'] + cl['w1'] * cl['E']
            eps2 = ch['w2'] * ch['E'] + cl['w2'] * cl['E']
            delta = (ch['E'] - cl['E']) / 2
            site = (eps1 - eps2) / 2
            if abs(delta) <= abs(site): continue
            J = float(np.sqrt(delta ** 2 - site ** 2))
            return J, ch['idx'], cl['idx'], candidates[:5]
    return np.nan, -1, -1, candidates


def reorder_indices(symbols):
    """Given the original (input) list of atom symbols, return the indices in
    the *DFTB+ internal* (element-grouped) order. DFTB+ writes the atom list
    in the order of TypeNames declaration; we declare H, C, N, Fe."""
    elem_order = ['H', 'C', 'N', 'Fe']
    return sorted(range(len(symbols)),
                  key=lambda i: (elem_order.index(symbols[i]), i))
