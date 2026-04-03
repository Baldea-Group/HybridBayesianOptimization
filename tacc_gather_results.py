"""
gather_results.py - Merge per-job pickle files into combined results.

Reads individual pickles from results_parallel/, groups by (n_init, xi),
and writes combined pickle files matching the format run_comparison.py expects.
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path


def gather(indir, outdir, problem_set='all'):
    indir = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(indir.glob('*.pkl'))
    if not pkl_files:
        print(f"No pickle files found in {indir}")
        return

    # Group files by (n_init, xi)
    groups = defaultdict(list)
    for f in pkl_files:
        with open(f, 'rb') as fh:
            data = pickle.load(fh)
        settings = data['settings']
        key = (settings['n_initial'], settings['xi'])
        groups[key].append(data)

    print(f"Found {len(pkl_files)} files in {len(groups)} (n_init, xi) groups")

    for (n_init, xi), data_list in sorted(groups.items()):
        # Merge all problem results into one dict
        merged_results = {}
        ref_settings = None
        for data in data_list:
            if ref_settings is None:
                ref_settings = data['settings'].copy()
                ref_settings['problem_names'] = []
            for pname, presults in data['results'].items():
                if pname in merged_results:
                    print(f"  Warning: duplicate problem {pname} for "
                          f"n_init={n_init}, xi={xi} -- skipping")
                    continue
                merged_results[pname] = presults
                if pname not in ref_settings['problem_names']:
                    ref_settings['problem_names'].append(pname)

        # Update settings with full problem list
        ref_settings['n_initial'] = n_init
        ref_settings['xi'] = xi

        # Build output filename matching run_comparison.py convention
        acq_str = '_'.join(ref_settings['acquisitions'])
        fname = f"results_{problem_set}_{acq_str}_ninit{n_init}_xi{xi}.pkl"
        out_path = outdir / fname

        with open(out_path, 'wb') as f:
            pickle.dump({'settings': ref_settings, 'results': merged_results}, f)
        print(f"  n_init={n_init}, xi={xi}: {len(merged_results)} problems -> {out_path}")

    print(f"\nDone. Combined results in {outdir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gather parallel job results into combined pickle files')
    parser.add_argument('--indir', type=str, default='results_parallel',
                        help='Directory with per-job pickle files')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Output directory for combined results')
    parser.add_argument('--problem_set', type=str, default='all',
                        help='Problem set name for output filename')
    args = parser.parse_args()

    gather(args.indir, args.outdir, args.problem_set)
