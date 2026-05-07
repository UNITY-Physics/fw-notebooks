#!/usr/bin/env python3
"""
download_derivatives_cli.py

Download and compile segmentation derivatives from Flywheel into a single CSV.

Usage examples:

  # Single project, minimorph + volume only
  python download_derivatives_cli.py \
    --projects "Project A" \
    --tools minimorph \
    --keywords volumes \
    --input-source MRR \
    --output-dir ./outputs

  # Multiple projects, recon-all-clinical (area + thickness + volume) + minimorph
  python download_derivatives_cli.py \
    --projects "Project A" "Project B" \
    --tools recon-all-clinical minimorph \
    --keywords area thickness volume volumes \
    --input-source MRR \
    --output-dir ./outputs \
    --fw-session-info

  # Debug mode: only process first 5 sessions per project
  python download_derivatives_cli.py \
    --projects "Project A" \
    --tools minimorph \
    --keywords volumes \
    --input-source MRR \
    --output-dir ./outputs \
    --debug

  # Use Gambas input instead of MRR
  python download_derivatives_cli.py \
    --projects "Project A" \
    --tools recon-all-clinical \
    --keywords volume \
    --input-source "Enhanced (Gambas)" \
    --output-dir ./outputs

Required environment variable:
  FW_CLI_API_KEY   Your Flywheel API key (or pass via --api-key)
"""

import os
import sys
import argparse
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import flywheel
import numpy as np
import pandas as pd
import pathvalidate as pv
import yaml
from packaging import version


# ---------------------------------------------------------------------------
# Core per-session logic (ported from Streamlit app, no st.* calls)
# ---------------------------------------------------------------------------

def get_latest(analyses, input_keyword):
    """Return list containing the most recently created analysis whose first
    input filename contains any of the given keywords."""
    kws = [input_keyword] if isinstance(input_keyword, str) else input_keyword
    candidates = [a for a in analyses if any(kw in a.inputs[0].name for kw in kws)]
    return [max(candidates, key=lambda a: a.created)] if candidates else []


def _build_session_df(fw, project, session, analyses_list,
                      fw_session_info, keywords, tool_map,
                      project_path):
    """
    Internal helper: download files for a pre-filtered list of analyses and
    merge them horizontally into a single DataFrame row.
    Returns an empty DataFrame if nothing matched.
    """
    ses_label = session.label
    sub_label = session.subject.label
    session_df = pd.DataFrame()

    if fw_session_info:
        session_tags = session.tags if session.tags else []
        session_df['session_tags'] = ' | '.join(session_tags) if session_tags else 'n/a'
        for key, value in session.info.items():
            session_df[key] = value

    for analysis in analyses_list:
        analysis = analysis.reload()
        gear = analysis.gear_info.name
        volumetric_cols = tool_map.get(gear, [])

        matched_files = [
            f for f in analysis.files
            if any(kw in f.name for kw in keywords)
        ]
        print(f"    gear={gear}: {len(matched_files)} file(s) matched")

        for analysis_file in matched_files:
            file = analysis_file.reload()
            download_dir = pv.sanitize_filepath(
                project_path / sub_label / ses_label, platform='auto'
            )
            download_dir.mkdir(parents=True, exist_ok=True)
            download_path = download_dir / file.name

            print(f"    Downloading {file.name} ...")
            file.download(download_path)

            df = pd.read_csv(download_path)
            df["project"]   = project.label
            df["subject"]   = sub_label
            df["sex"]       = session.info.get('childBiologicalSex', 'n/a')
            df["session"]   = ses_label
            df["childTimepointAge_months"] = session.info.get(
                'childTimepointAge_months', df.get("age", "n/a")
            )

            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.insert(3, 'session_qc', session.tags[-1] if session.tags else 'n/a')
            df["age_source"] = "custom_info"

            # Prefix volumetric columns by gear
            if gear == "minimorph":
                df["analysis_id_mm"]    = analysis.id
                df["gear_v_minimorph"]  = analysis.gear_info.version
                df.rename(columns={c: f'mm_{c}' for c in volumetric_cols if c in df.columns}, inplace=True)
            elif gear == "supersynth":
                df["analysis_id_ss"]     = analysis.id
                df["gear_v_supersynth"]  = analysis.gear_info.version
                df.rename(columns={c: f'ss_{c}' for c in volumetric_cols if c in df.columns}, inplace=True)
            else:  # recon-all-clinical / recon-any
                df["analysis_id_ra"]    = analysis.id
                df["gear_v_recon_all"]  = analysis.gear_info.version
                df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
                volumetric_cols_norm = [c.replace(' ', '_').replace('-', '_').lower() for c in volumetric_cols]
                df.rename(columns={c: f'ra_{c}' for c in volumetric_cols_norm if c in df.columns}, inplace=True)

            # Merge horizontally into session_df
            if session_df.empty:
                session_df = df
            else:
                merge_keys = ['subject', 'session']
                cols_to_exclude = [c for c in session_df.columns if c not in merge_keys]
                new_cols = [c for c in df.columns if c not in cols_to_exclude]
                session_df = pd.merge(
                    session_df, df[new_cols],
                    on=merge_keys, how='outer'
                )

            os.remove(download_path)

    session_df.drop(columns=['gear_v', 'age_source', 'template_age'],
                    inplace=True, errors='ignore')
    return session_df


def download_session_data(fw, project, session_id, project_path,
                          segtool, input_source, fw_session_info,
                          keywords, tool_map):
    """
    Download all matching CSV files for one session and merge them
    horizontally into a single DataFrame row.

    Returns:
      - input_source == "Both": dict {'mrr': df_or_None, 'gambas': df_or_None}
      - otherwise:              a single DataFrame, or None
    """
    session = fw.get(session_id).reload()
    ses_label = session.label
    sub_label = session.subject.label

    # ---- Find completed analyses for the requested gears ----
    analyses = [
        a for a in session.analyses
        if a.reload().gear_info is not None
        and a.reload().gear_info.name in segtool
        and a.reload().job.get('state') == 'complete'
    ]

    mrr_analyses = []
    gambas_analyses = []

    for segmentation_tool in segtool:
        tool_analyses = [a for a in analyses if a.gear_info.name == segmentation_tool]

        if input_source in ("MRR", "Both"):
            mrr_analyses.extend(get_latest(tool_analyses, "mrr"))
        if input_source in ("Enhanced (Gambas)", "Both"):
            gambas_analyses.extend(get_latest(tool_analyses, ["gambas", "ResCNN"]))

    if input_source == "Both":
        print(f"  [{sub_label} / {ses_label}] "
              f"analyses found: {len(analyses)}, "
              f"MRR: {len(mrr_analyses)}, Gambas: {len(gambas_analyses)}")
    else:
        combined_count = len(mrr_analyses) + len(gambas_analyses)
        print(f"  [{sub_label} / {ses_label}] "
              f"analyses found: {len(analyses)}, after filtering: {combined_count}")

    try:
        if input_source == "Both":
            mrr_df = _build_session_df(
                fw, project, session, mrr_analyses,
                fw_session_info, keywords, tool_map, project_path
            )
            gambas_df = _build_session_df(
                fw, project, session, gambas_analyses,
                fw_session_info, keywords, tool_map, project_path
            )
            return {
                'mrr':    mrr_df    if not mrr_df.empty    else None,
                'gambas': gambas_df if not gambas_df.empty else None,
            }
        else:
            analyses_filtered = mrr_analyses + gambas_analyses
            session_df = _build_session_df(
                fw, project, session, analyses_filtered,
                fw_session_info, keywords, tool_map, project_path
            )
            return session_df if not session_df.empty else None

    except Exception:
        print(f"  EXCEPTION [{sub_label} / {ses_label}]:\n{traceback.format_exc()}")
        if input_source == "Both":
            return {'mrr': None, 'gambas': None}
        return None


# ---------------------------------------------------------------------------
# Per-project orchestrator
# ---------------------------------------------------------------------------

def download_derivatives(fw, project_id, segtool, input_source,
                         fw_session_info, keywords, data_dir,
                         tool_map, debug=False):
    project = fw.projects.find_first(f'label={project_id}')
    if project is None:
        print(f"[WARNING] Project '{project_id}' not found — skipping.")
        return None

    print(f"\n{'='*60}")
    print(f"Project : {project.label}")
    print(f"Subjects: {len(project.subjects())}")
    print(f"Sessions: {len(project.sessions())}")
    print(f"{'='*60}")

    project_path = pv.sanitize_filepath(data_dir / project.label, platform='auto')
    project_path.mkdir(parents=True, exist_ok=True)

    sessions = [s.id for s in project.sessions()
                if not s.subject.label.startswith('137-')]
    if debug:
        sessions = sessions[:5]
        print(f"[DEBUG] Limiting to first {len(sessions)} sessions.")

    mrr_frames    = []
    gambas_frames = []
    all_frames    = []  # used for single-source modes

    for i, session_id in enumerate(sessions, 1):
        print(f"\n[{i}/{len(sessions)}] Processing session {session_id} ...")
        result = download_session_data(
            fw, project, session_id, project_path,
            segtool, input_source, fw_session_info,
            keywords, tool_map
        )

        if input_source == "Both":
            # result is a dict {'mrr': df_or_None, 'gambas': df_or_None}
            if result['mrr'] is not None:
                mrr_frames.append(result['mrr'])
            if result['gambas'] is not None:
                gambas_frames.append(result['gambas'])
            n_mrr    = len(mrr_frames)
            n_gambas = len(gambas_frames)
            print(f"  -> MRR frames so far: {n_mrr}, Gambas frames so far: {n_gambas}")
        else:
            if result is not None:
                all_frames.append(result)
                print(f"  -> appended, frames so far: {len(all_frames)}")
            else:
                print(f"  -> no data returned, skipping.")

    def _finalise(frames, label_suffix):
        """Dedup, reorder, save and return path for one frame list."""
        if not frames:
            print(f"[WARNING] No {label_suffix} results for project '{project_id}'.")
            return None
        combined = pd.concat(frames, ignore_index=True)

        for gear_col in ['gear_v_recon_all', 'gear_v_minimorph']:
            if gear_col in combined.columns:
                key_cols = [c for c in ['subject', 'session', 'acquisition'] if c in combined.columns]
                combined = (
                    combined
                    .sort_values(
                        gear_col,
                        key=lambda s: s.map(
                            lambda v: version.parse(v) if pd.notna(v) else version.parse("0")
                        ),
                        ascending=False
                    )
                    .drop_duplicates(subset=key_cols, keep='first')
                )

        for col in ['scanner_software_v', 'input_gear_v']:
            if col in combined.columns and 'acquisition' in combined.columns:
                cols = combined.columns.tolist()
                cols.insert(cols.index('acquisition') + 1, cols.pop(cols.index(col)))
                combined = combined[cols]

        segtool_str = '-'.join(segtool)
        outname = project.label.replace(' ', '_').replace('(', '').replace(')', '')
        outpath = project_path / f"{outname}-{segtool_str}-{label_suffix}.csv"
        combined.to_csv(outpath, index=False)
        print(f"\n[OK] Saved {len(combined)} rows ({label_suffix}) to {outpath}")
        return str(outpath)

    if input_source == "Both":
        mrr_path    = _finalise(mrr_frames,    "MRR")
        gambas_path = _finalise(gambas_frames, "Gambas")
        # Return as a tuple so the caller can handle both streams
        return (mrr_path, gambas_path)
    else:
        return _finalise(all_frames, input_source.replace(' ', '_').replace('(', '').replace(')', ''))


# ---------------------------------------------------------------------------
# Final assembly across projects
# ---------------------------------------------------------------------------

def assemble_csv(derivative_paths, output_dir, label=""):
    frames = []
    for path in derivative_paths:
        df_proj = pd.read_csv(path)
        frames.append(df_proj)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined.drop(columns=['age', 'sex', 'gear_v'], inplace=True, errors='ignore')

    front_cols = [
        'project', 'subject', 'session', 'childTimepointAge_months',
        'childBiologicalSex', 'studyTimepoint', 'session_qc', 'acquisition'
    ]
    cols = combined.columns.tolist()
    ra_cols   = [c for c in cols if c.startswith('ra_')]
    mm_cols   = [c for c in cols if c.startswith('mm_')]
    ss_cols   = [c for c in cols if c.startswith('ss_')]
    spoken_for = set(front_cols + ra_cols + mm_cols + ss_cols)
    other_cols = [c for c in cols if c not in spoken_for]

    new_order = front_cols + ra_cols + mm_cols + ss_cols + other_cols

    for col in front_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined[new_order]

    unique_projects = combined["project"].dropna().unique()
    project_str = '_'.join(unique_projects)
    time_str    = datetime.now().strftime("%Y%m%d-%H%M%S")
    label_str   = f"_{label}" if label else ""
    out_csv     = Path(output_dir) / f"derivatives_summary_{project_str}{label_str}_{time_str}.csv"
    combined.to_csv(out_csv, index=False)
    return combined, str(out_csv)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Flywheel segmentation derivatives to a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--projects', nargs='+', required=True, metavar='PROJECT',
        help='One or more Flywheel project labels (quote labels with spaces).'
    )
    parser.add_argument(
        '--tools', nargs='+', required=True,
        choices=['recon-all-clinical', 'minimorph', 'supersynth', 'recon-any'],
        metavar='TOOL',
        help='Segmentation gear(s) to pull results from.'
    )
    parser.add_argument(
        '--keywords', nargs='+', required=True, metavar='KEYWORD',
        help='Filename keywords to match (e.g. volume area thickness volumes).'
    )
    parser.add_argument(
        '--input-source', default='MRR',
        choices=['MRR', 'Enhanced (Gambas)', 'Both'],
        help='Which structural input to target (default: MRR).'
    )
    parser.add_argument(
        '--output-dir', default='./outputs', metavar='DIR',
        help='Directory to save output CSVs (default: ./outputs).'
    )
    parser.add_argument(
        '--vol-columns-yml', default=None, metavar='PATH',
        help='Path to vol_columns.yml. Defaults to utils/vol_columns.yml '
             'relative to this script.'
    )
    parser.add_argument(
        '--api-key', default=None,
        help='Flywheel API key. Falls back to FW_CLI_API_KEY env var.'
    )
    parser.add_argument(
        '--fw-session-info', action='store_true',
        help='Include Flywheel session tags and custom info in output.',default=True,
    )
    parser.add_argument(
        '--max-workers', type=int, default=4, metavar='N',
        help='Number of projects to process in parallel (default: 4).'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Only process first 5 sessions per project (for testing).'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- API key ----
    api_key = args.api_key or os.getenv('FW_CLI_API_KEY')
    if not api_key:
        sys.exit("[ERROR] No Flywheel API key found. "
                 "Set FW_CLI_API_KEY or use --api-key.")
    fw = flywheel.Client(api_key)
    print(f"Connected to Flywheel: {fw.get_config().site.get('api_url', '?')}")

    # ---- vol_columns.yml ----
    if args.vol_columns_yml:
        yml_path = Path(args.vol_columns_yml)
    else:
        yml_path = Path(__file__).parent.parent / "utils" / "vol_columns.yml"

    if not yml_path.exists():
        sys.exit(f"[ERROR] vol_columns.yml not found at {yml_path}. "
                 "Use --vol-columns-yml to specify its location.")

    with open(yml_path) as f:
        tool_map = yaml.load(f, Loader=yaml.SafeLoader)

    # ---- Output directory ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / 'session_data'
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSettings:")
    print(f"  Projects     : {args.projects}")
    print(f"  Tools        : {args.tools}")
    print(f"  Keywords     : {args.keywords}")
    print(f"  Input source : {args.input_source}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Session info : {args.fw_session_info}")
    print(f"  Debug mode   : {args.debug}")

    # ---- Thread-local Flywheel clients (one per worker thread) ----
    _thread_local = threading.local()

    def _get_fw():
        if not hasattr(_thread_local, 'fw'):
            _thread_local.fw = flywheel.Client(api_key)
        return _thread_local.fw

    def _run_project(proj):
        return download_derivatives(
            fw=_get_fw(),
            project_id=proj,
            segtool=args.tools,
            input_source=args.input_source,
            fw_session_info=args.fw_session_info,
            keywords=args.keywords,
            data_dir=data_dir,
            tool_map=tool_map,
            debug=args.debug
        )

    # ---- Run per project (parallelised) ----
    mrr_paths    = []
    gambas_paths = []
    single_paths = []

    n_workers = min(args.max_workers, len(args.projects))
    print(f"\nProcessing {len(args.projects)} project(s) with {n_workers} worker(s) ...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_proj = {executor.submit(_run_project, proj): proj
                          for proj in args.projects}
        for future in as_completed(future_to_proj):
            proj = future_to_proj[future]
            try:
                result = future.result()
            except Exception:
                print(f"[ERROR] Project '{proj}' raised an exception:\n"
                      f"{traceback.format_exc()}")
                continue

            if args.input_source == "Both":
                if result is None:
                    continue
                mrr_path, gambas_path = result
                if mrr_path:
                    mrr_paths.append(mrr_path)
                if gambas_path:
                    gambas_paths.append(gambas_path)
            else:
                if result:
                    single_paths.append(result)

    if args.input_source == "Both":
        if not mrr_paths and not gambas_paths:
            sys.exit("[ERROR] No derivatives found across any project.")

        print("\nAssembling MRR CSV ...")
        if mrr_paths:
            mrr_combined, mrr_csv = assemble_csv(mrr_paths, output_dir, label="MRR")
            print(f"\n[DONE] MRR CSV  : {mrr_csv}")
            print(f"       Shape     : {mrr_combined.shape}")
        else:
            print("[WARNING] No MRR data to assemble.")

        print("\nAssembling Gambas CSV ...")
        if gambas_paths:
            gambas_combined, gambas_csv = assemble_csv(gambas_paths, output_dir, label="Gambas")
            print(f"\n[DONE] Gambas CSV: {gambas_csv}")
            print(f"       Shape      : {gambas_combined.shape}")
        else:
            print("[WARNING] No Gambas data to assemble.")

    else:
        if not single_paths:
            sys.exit("[ERROR] No derivatives found across any project.")

        # ---- Assemble final CSV ----
        print("\nAssembling final CSV ...")
        combined, out_csv = assemble_csv(single_paths, output_dir)
        print(f"\n[DONE] Final CSV: {out_csv}")
        print(f"       Shape    : {combined.shape}")


if __name__ == '__main__':
    main()
