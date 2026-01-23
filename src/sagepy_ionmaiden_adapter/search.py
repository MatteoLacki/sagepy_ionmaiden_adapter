import argparse
import mmappet
import multiprocessing
import numpy as np
import pandas as pd
import toml

from dictodot import DotDict
from pathlib import Path
from tqdm import tqdm

from sagepy.core import EnzymeBuilder
from sagepy.core import Precursor
from sagepy.core import RawSpectrum
from sagepy.core import SageSearchConfiguration
from sagepy.core import Scorer
from sagepy.core import SpectrumProcessor
from sagepy.core import Tolerance
from sagepy.core.fdr import sage_fdr_psm
from sagepy.core.scoring import ScoreType
from sagepy.utility import compress_psms
from sagepy.utility import psm_collection_to_pandas

# import json
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", 5)

# pmsms_path = Path("temp/F9477/correlation/pmsms.mmappet")
# fasta_path = "fastas/Human_2024_02_16_UniProt_Taxon9606_Reviewed_20434entries_contaminant_tenzer.fasta"
# max_batch_size = 100_000
# num_threads = multiprocessing.cpu_count()
# config_path = "configs/sagepy/devel.toml"
# output_folder = Path("/home/matteo/tmp/sagepy_psms")

# with open("configs/sage/default.json", "r") as f:
#     default_config = DotDict.Recursive(json.load(f))


def get_raw_spectrum(precursors, fragments, prec_idx):
    P = precursors.iloc[prec_idx]
    precursor = Precursor(charge=None, mz=P.mz)
    start = P.fragment_spectrum_start
    diff = P.fragment_event_cnt
    mzs = fragments["mz"][start : start + diff]
    intensities = fragments["intensity"][start : start + diff]
    raw_spectrum = RawSpectrum(
        file_id=1,
        spec_id=str(prec_idx),
        total_ion_current=intensities.sum(),
        precursors=[precursor],
        mz=mzs,
        intensity=intensities,
        scan_start_time=P.rt,
        ion_injection_time=P.rt,
        mobility=np.array([P.inv_ion_mobility]),
    )
    return raw_spectrum


def batch_ranges(N, K):
    for start in range(0, N, K):
        end = min(start + K, N)
        yield start, end


def search_data(
    output_folder: Path,
    pmsms_path: Path,
    fasta_path: Path,
    config_path: Path,
    max_batch_size: int = 100_000,
    num_threads: int = multiprocessing.cpu_count(),
):
    with open(config_path, "r") as f:
        config = DotDict.Recursive(toml.load(f))

    enzyme_builder = EnzymeBuilder(
        missed_cleavages=config.database.enzyme.missed_cleavages,
        min_len=config.database.enzyme.min_len,
        max_len=config.database.enzyme.max_len,
        cleave_at=config.database.enzyme.cleave_at,
        restrict=config.database.enzyme.restrict,
        c_terminal=config.database.enzyme.c_terminal,
        semi_enzymatic=None,
    )

    with open(fasta_path, "r") as infile:
        fasta = infile.read()

    # U -> UNIMOD translation
    static_mods = {
        k: v.replace("U", "UNIMOD")
        for k, v in config.database.enzyme.static_mods.items()
    }
    variable_mods = {
        k: [v.replace("U", "UNIMOD") for v in vv]
        for k, vv in config.database.enzyme.variable_mods.items()
    }

    sage_config = SageSearchConfiguration(
        fasta=fasta,
        static_mods=static_mods,
        variable_mods=variable_mods,
        enzyme_builder=enzyme_builder,
        generate_decoys=config.database.generate_decoys,
        bucket_size=config.database.bucket_size,
    )

    indexed_db = sage_config.generate_indexed_database()
    precursors = pd.read_parquet(pmsms_path / "precursors.parquet")
    fragments = mmappet.open_dataset_dct(pmsms_path)

    spec_processor = SpectrumProcessor(
        take_top_n=max_batch_size,
        deisotope=config.deisotope,
    )

    scorer = Scorer(
        precursor_tolerance=Tolerance(
            **{k: tuple(v) for k, v in config.precursor_tol.items()}
        ),
        fragment_tolerance=Tolerance(
            **{k: tuple(v) for k, v in config.fragment_tol.items()}
        ),
        min_matched_peaks=config.min_matched_peaks,
        min_isotope_err=config.isotope_errors[0],
        max_isotope_err=config.isotope_errors[1],
        min_precursor_charge=config.precursor_charge[0],
        max_precursor_charge=config.precursor_charge[1],
        chimera=config.chimera,
        report_psms=config.report_psms,
        wide_window=config.wide_window,
        annotate_matches=True,
        override_precursor_charge=False,
        score_type=ScoreType(config.score_type),
        max_fragment_charge=config.max_fragment_charge,
        variable_mods=variable_mods,
        static_mods=static_mods,
    )

    results_features = {}
    with tqdm(total=len(precursors), desc="Querying spectra") as pbar:
        # for start, end in batch_ranges(10_000, max_batch_size):
        for start, end in batch_ranges(len(precursors), max_batch_size):
            query_batch = [
                spec_processor.process(
                    get_raw_spectrum(precursors, fragments, prec_idx)
                )
                for prec_idx in range(start, end)
            ]
            results_features.update(
                scorer.score_collection_psm(
                    db=indexed_db,
                    spectrum_collection=query_batch,
                    num_threads=num_threads,
                )
            )
            pbar.update(end - start)

    # This function changes the PSMs in-place, meaning the so-far 1.0 defaults q-values are replaced with the SAGE calculated ones

    # temporary fix: until David fixes ion_mobility internally.
    for prec_id_str, psms in results_features.items():
        prec_id = int(prec_id_str)
        for psm in psms:
            psm.inverse_ion_mobility = precursors.inv_ion_mobility.iloc[prec_id]

    sage_fdr_psm(results_features, indexed_db, use_hyper_score=True)
    PSM_pandas_features = psm_collection_to_pandas(results_features)
    psm_list = [v for vv in results_features.values() for v in vv]

    output_folder.mkdir(parents=True, exist_ok=True)
    PSM_pandas_features.to_parquet(output_folder / "findings.parquet")
    with open(output_folder / "psms.bin", "wb") as outfile:
        outfile.write(compress_psms(psm_list))


def cli():
    parser = argparse.ArgumentParser(
        description="Search data using MS/MS spectra and a FASTA database."
    )

    # Required positional arguments
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Folder where output files will be written",
    )
    parser.add_argument(
        "pmsms_path",
        type=Path,
        help="Path to the pMSMS input file",
    )
    parser.add_argument(
        "fasta_path",
        type=Path,
        help="Path to the FASTA database file",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the configuration file",
    )

    # Optional arguments
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=100_000,
        help="Maximum number of items processed per batch (default: 100000)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help=(
            "Number of worker threads to use "
            f"(default: number of CPUs = {multiprocessing.cpu_count()})"
        ),
    )
    args = parser.parse_args()
    search_data(**args.__dict__)
