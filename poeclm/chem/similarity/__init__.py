# ruff: noqa: E402
"""Similarity search module.

Based on the code from Andrew Dalke's blog post:
    http://www.dalkescientific.com/writings/diary/archive/2020/10/07/intersection_popcount.html
"""

import hashlib
import os
from pathlib import Path

from cffi import FFI

parent = Path(__file__).parent

ffibuilder = FFI()
ffibuilder.cdef(
    """
int byte_popcount_512(const unsigned char *fp);
int byte_popcount_1024(const unsigned char *fp);
int byte_popcount_2048(const unsigned char *fp);

int byte_intersect_512(const unsigned char *fp1, const unsigned char *fp2);
int byte_intersect_1024(const unsigned char *fp1, const unsigned char *fp2);
int byte_intersect_2048(const unsigned char *fp1, const unsigned char *fp2);

double byte_tanimoto_512(const unsigned char *fp1, const unsigned char *fp2);
double byte_tanimoto_1024(const unsigned char *fp1, const unsigned char *fp2);
double byte_tanimoto_2048(const unsigned char *fp1, const unsigned char *fp2);

int threshold_bin_tanimoto_search_512(const unsigned char *query_fp,
                                      const int query_popcount,
                                      const double threshold,
                                      const int num_targets,
                                      const unsigned char *target_fps,
                                      const int target_popcount,
                                      int *hit_indices,
                                      double *hit_scores);
int threshold_bin_tanimoto_search_1024(const unsigned char *query_fp,
                                       const int query_popcount,
                                       const double threshold,
                                       const int num_targets,
                                       const unsigned char *target_fps,
                                       const int target_popcount,
                                       int *hit_indices,
                                       double *hit_scores);
int threshold_bin_tanimoto_search_2048(const unsigned char *query_fp,
                                       const int query_popcount,
                                       const double threshold,
                                       const int num_targets,
                                       const unsigned char *target_fps,
                                       const int target_popcount,
                                       int *hit_indices,
                                       double *hit_scores);

bool threshold_bin_tanimoto_hit_512(const unsigned char *query_fp,
                                    const int query_popcount,
                                    const double threshold,
                                    const int num_targets,
                                    const unsigned char *target_fps,
                                    const int target_popcount);
bool threshold_bin_tanimoto_hit_1024(const unsigned char *query_fp,
                                     const int query_popcount,
                                     const double threshold,
                                     const int num_targets,
                                     const unsigned char *target_fps,
                                     const int target_popcount);
bool threshold_bin_tanimoto_hit_2048(const unsigned char *query_fp,
                                     const int query_popcount,
                                     const double threshold,
                                     const int num_targets,
                                     const unsigned char *target_fps,
                                     const int target_popcount);

void topk_tanimoto_search_512(const unsigned char *query_fp,
                              const int k,
                              const int num_targets,
                              const unsigned char *target_fps,
                              int *topk_indices,
                              double *topk_scores);
void topk_tanimoto_search_1024(const unsigned char *query_fp,
                               const int k,
                               const int num_targets,
                               const unsigned char *target_fps,
                               int *topk_indices,
                               double *topk_scores);
void topk_tanimoto_search_2048(const unsigned char *query_fp,
                               const int k,
                               const int num_targets,
                               const unsigned char *target_fps,
                               int *topk_indices,
                               double *topk_scores);
"""
)
c_path = parent / "popc.c"
with c_path.open() as f:
    c_source = f.read()
ffibuilder.set_source(
    "_popc",
    c_source,
    extra_compile_args=["-mpopcnt", "-O3", "-march=native"],
)

python_path = Path(__file__)
with python_path.open() as f:
    python_source = f.read()
source_hash = hashlib.md5(c_source.encode() + python_source.encode()).hexdigest()
hash_path = parent / "_popc.hash"
build = True
if hash_path.exists():
    with hash_path.open() as f:
        if source_hash == f.read():
            build = False

if build:
    prev_cwd = Path.cwd()
    os.chdir(parent)
    ffibuilder.compile()
    os.chdir(prev_cwd)
    with hash_path.open("w") as f:
        f.write(source_hash)

from .threshold_search import ThresholdTanimotoSearch
from .topk_search import TopKTanimotoSearch

__all__ = [
    "ThresholdTanimotoSearch",
    "TopKTanimotoSearch",
]
