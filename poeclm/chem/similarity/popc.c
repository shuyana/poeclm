/* Based on the code from Andrew Dalke's blog post:
    http://www.dalkescientific.com/writings/diary/archive/2020/10/07/intersection_popcount.html
*/
#include <stdbool.h>
#include <stdint.h>

static int byte_popcount_512(const unsigned char *fp)
{
    const uint64_t *fp_64 = (uint64_t *)fp;
    const int num_words = 512 / 64;

    int popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        popcount += __builtin_popcountll(fp_64[i]);
    }
    return popcount;
}

static int byte_popcount_1024(const unsigned char *fp)
{
    const uint64_t *fp_64 = (uint64_t *)fp;
    const int num_words = 1024 / 64;

    int popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        popcount += __builtin_popcountll(fp_64[i]);
    }
    return popcount;
}

static int byte_popcount_2048(const unsigned char *fp)
{
    const uint64_t *fp_64 = (uint64_t *)fp;
    const int num_words = 2048 / 64;

    int popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        popcount += __builtin_popcountll(fp_64[i]);
    }
    return popcount;
}

static int byte_intersect_512(const unsigned char *fp1, const unsigned char *fp2)
{
    const uint64_t *fp1_64 = (uint64_t *)fp1, *fp2_64 = (uint64_t *)fp2;
    const int num_words = 512 / 64;

    int intersect_popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        intersect_popcount += __builtin_popcountll(fp1_64[i] & fp2_64[i]);
    }
    return intersect_popcount;
}

static int byte_intersect_1024(const unsigned char *fp1, const unsigned char *fp2)
{
    const uint64_t *fp1_64 = (uint64_t *)fp1, *fp2_64 = (uint64_t *)fp2;
    const int num_words = 1024 / 64;

    int intersect_popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        intersect_popcount += __builtin_popcountll(fp1_64[i] & fp2_64[i]);
    }
    return intersect_popcount;
}

static int byte_intersect_2048(const unsigned char *fp1, const unsigned char *fp2)
{
    const uint64_t *fp1_64 = (uint64_t *)fp1, *fp2_64 = (uint64_t *)fp2;
    const int num_words = 2048 / 64;

    int intersect_popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        intersect_popcount += __builtin_popcountll(fp1_64[i] & fp2_64[i]);
    }
    return intersect_popcount;
}

static double byte_tanimoto_512(const unsigned char *fp1, const unsigned char *fp2)
{
    const uint64_t *fp1_64 = (uint64_t *)fp1, *fp2_64 = (uint64_t *)fp2;
    const int num_words = 512 / 64;

    int union_popcount = 0, intersect_popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        intersect_popcount += __builtin_popcountll(fp1_64[i] & fp2_64[i]);
        union_popcount += __builtin_popcountll(fp1_64[i] | fp2_64[i]);
    }
    if (union_popcount == 0)
    {
        return 0.0;
    }
    return ((double)intersect_popcount) / union_popcount;
}

static double byte_tanimoto_1024(const unsigned char *fp1, const unsigned char *fp2)
{
    const uint64_t *fp1_64 = (uint64_t *)fp1, *fp2_64 = (uint64_t *)fp2;
    const int num_words = 1024 / 64;

    int union_popcount = 0, intersect_popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        intersect_popcount += __builtin_popcountll(fp1_64[i] & fp2_64[i]);
        union_popcount += __builtin_popcountll(fp1_64[i] | fp2_64[i]);
    }
    if (union_popcount == 0)
    {
        return 0.0;
    }
    return ((double)intersect_popcount) / union_popcount;
}

static double byte_tanimoto_2048(const unsigned char *fp1, const unsigned char *fp2)
{
    const uint64_t *fp1_64 = (uint64_t *)fp1, *fp2_64 = (uint64_t *)fp2;
    const int num_words = 2048 / 64;

    int union_popcount = 0, intersect_popcount = 0;
    for (int i = 0; i < num_words; i++)
    {
        intersect_popcount += __builtin_popcountll(fp1_64[i] & fp2_64[i]);
        union_popcount += __builtin_popcountll(fp1_64[i] | fp2_64[i]);
    }
    if (union_popcount == 0)
    {
        return 0.0;
    }
    return ((double)intersect_popcount) / union_popcount;
}

static int threshold_bin_tanimoto_search_512(const unsigned char *query_fp,
                                             const int query_popcount,
                                             const double threshold,
                                             const int num_targets,
                                             const unsigned char *target_fps,
                                             const int target_popcount,
                                             int *hit_indices,
                                             double *hit_scores)
{
    const double total_popcount = (double)(query_popcount + target_popcount);
    const int num_bytes = 512 / 8;

    /* Handle edge cases if one or both of the fingerprints has no bits set. */
    if ((threshold > 0.0) && ((query_popcount == 0) || (target_popcount == 0)))
    {
        /* Nothing can match: Tanimoto(0, x) = 0 < threshold */
        return 0;
    }
    else if ((threshold <= 0.0) && (total_popcount == 0.0))
    {
        /* Everything will match: Tanimoto(0, 0) = 0 >= threshold */
        for (int target_idx = 0; target_idx < num_targets; target_idx++)
        {
            hit_indices[target_idx] = target_idx;
            hit_scores[target_idx] = 0.0;
        }
        return num_targets;
    }

    /* Do the search */
    int num_hits = 0;
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const int intersect_popcount = byte_intersect_512(query_fp, target_fps + target_idx * num_bytes);
        const double score = intersect_popcount / (total_popcount - intersect_popcount);
        if (score >= threshold)
        {
            hit_indices[num_hits] = target_idx;
            hit_scores[num_hits] = score;
            num_hits++;
        }
    }
    return num_hits;
}

static int threshold_bin_tanimoto_search_1024(const unsigned char *query_fp,
                                              const int query_popcount,
                                              const double threshold,
                                              const int num_targets,
                                              const unsigned char *target_fps,
                                              const int target_popcount,
                                              int *hit_indices,
                                              double *hit_scores)
{
    const double total_popcount = (double)(query_popcount + target_popcount);
    const int num_bytes = 1024 / 8;

    /* Handle edge cases if one or both of the fingerprints has no bits set. */
    if ((threshold > 0.0) && ((query_popcount == 0) || (target_popcount == 0)))
    {
        /* Nothing can match: Tanimoto(0, x) = 0 < threshold */
        return 0;
    }
    else if ((threshold <= 0.0) && (total_popcount == 0.0))
    {
        /* Everything will match: Tanimoto(0, 0) = 0 >= threshold */
        for (int target_idx = 0; target_idx < num_targets; target_idx++)
        {
            hit_indices[target_idx] = target_idx;
            hit_scores[target_idx] = 0.0;
        }
        return num_targets;
    }

    /* Do the search */
    int num_hits = 0;
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const int intersect_popcount = byte_intersect_1024(query_fp, target_fps + target_idx * num_bytes);
        const double score = intersect_popcount / (total_popcount - intersect_popcount);
        if (score >= threshold)
        {
            hit_indices[num_hits] = target_idx;
            hit_scores[num_hits] = score;
            num_hits++;
        }
    }
    return num_hits;
}

static int threshold_bin_tanimoto_search_2048(const unsigned char *query_fp,
                                              const int query_popcount,
                                              const double threshold,
                                              const int num_targets,
                                              const unsigned char *target_fps,
                                              const int target_popcount,
                                              int *hit_indices,
                                              double *hit_scores)
{
    const double total_popcount = (double)(query_popcount + target_popcount);
    const int num_bytes = 2048 / 8;

    /* Handle edge cases if one or both of the fingerprints has no bits set. */
    if ((threshold > 0.0) && ((query_popcount == 0) || (target_popcount == 0)))
    {
        /* Nothing can match: Tanimoto(0, x) = 0 < threshold */
        return 0;
    }
    else if ((threshold <= 0.0) && (total_popcount == 0.0))
    {
        /* Everything will match: Tanimoto(0, 0) = 0 >= threshold */
        for (int target_idx = 0; target_idx < num_targets; target_idx++)
        {
            hit_indices[target_idx] = target_idx;
            hit_scores[target_idx] = 0.0;
        }
        return num_targets;
    }

    /* Do the search */
    int num_hits = 0;
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const int intersect_popcount = byte_intersect_2048(query_fp, target_fps + target_idx * num_bytes);
        const double score = intersect_popcount / (total_popcount - intersect_popcount);
        if (score >= threshold)
        {
            hit_indices[num_hits] = target_idx;
            hit_scores[num_hits] = score;
            num_hits++;
        }
    }
    return num_hits;
}

static bool threshold_bin_tanimoto_hit_512(const unsigned char *query_fp,
                                           const int query_popcount,
                                           const double threshold,
                                           const int num_targets,
                                           const unsigned char *target_fps,
                                           const int target_popcount)
{
    const double total_popcount = (double)(query_popcount + target_popcount);
    const int num_bytes = 512 / 8;

    /* Handle edge cases if one or both of the fingerprints has no bits set. */
    if ((threshold > 0.0) && ((query_popcount == 0) || (target_popcount == 0)))
    {
        /* Nothing can match: Tanimoto(0, x) = 0 < threshold */
        return false;
    }
    else if ((threshold <= 0.0) && (total_popcount == 0.0))
    {
        /* Everything will match: Tanimoto(0, 0) = 0 >= threshold */
        return true;
    }

    /* Do the search */
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const int intersect_popcount = byte_intersect_512(query_fp, target_fps + target_idx * num_bytes);
        const double score = intersect_popcount / (total_popcount - intersect_popcount);
        if (score >= threshold)
        {
            return true;
        }
    }
    return false;
}

static bool threshold_bin_tanimoto_hit_1024(const unsigned char *query_fp,
                                            const int query_popcount,
                                            const double threshold,
                                            const int num_targets,
                                            const unsigned char *target_fps,
                                            const int target_popcount)
{
    const double total_popcount = (double)(query_popcount + target_popcount);
    const int num_bytes = 1024 / 8;

    /* Handle edge cases if one or both of the fingerprints has no bits set. */
    if ((threshold > 0.0) && ((query_popcount == 0) || (target_popcount == 0)))
    {
        /* Nothing can match: Tanimoto(0, x) = 0 < threshold */
        return false;
    }
    else if ((threshold <= 0.0) && (total_popcount == 0.0))
    {
        /* Everything will match: Tanimoto(0, 0) = 0 >= threshold */
        return true;
    }

    /* Do the search */
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const int intersect_popcount = byte_intersect_1024(query_fp, target_fps + target_idx * num_bytes);
        const double score = intersect_popcount / (total_popcount - intersect_popcount);
        if (score >= threshold)
        {
            return true;
        }
    }
    return false;
}

static bool threshold_bin_tanimoto_hit_2048(const unsigned char *query_fp,
                                            const int query_popcount,
                                            const double threshold,
                                            const int num_targets,
                                            const unsigned char *target_fps,
                                            const int target_popcount)
{
    const double total_popcount = (double)(query_popcount + target_popcount);
    const int num_bytes = 2048 / 8;

    /* Handle edge cases if one or both of the fingerprints has no bits set. */
    if ((threshold > 0.0) && ((query_popcount == 0) || (target_popcount == 0)))
    {
        /* Nothing can match: Tanimoto(0, x) = 0 < threshold */
        return false;
    }
    else if ((threshold <= 0.0) && (total_popcount == 0.0))
    {
        /* Everything will match: Tanimoto(0, 0) = 0 >= threshold */
        return true;
    }

    /* Do the search */
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const int intersect_popcount = byte_intersect_2048(query_fp, target_fps + target_idx * num_bytes);
        const double score = intersect_popcount / (total_popcount - intersect_popcount);
        if (score >= threshold)
        {
            return true;
        }
    }
    return false;
}

static void topk_tanimoto_search_512(const unsigned char *query_fp,
                                     const int k,
                                     const int num_targets,
                                     const unsigned char *target_fps,
                                     int *topk_indices,
                                     double *topk_scores)
{
    const int num_bytes = 512 / 8;

    /* Initialize the binary heap */
    for (int i = 0; i < k; i++)
    {
        topk_indices[i] = -1;
        topk_scores[i] = -1.0;
    }

    /* Do the search */
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const double score = byte_tanimoto_512(query_fp, target_fps + target_idx * num_bytes);

        /* If the score is greater than the smallest score in the heap */
        if (score > topk_scores[0])
        {
            /* Replace the root of the heap with the new node */
            topk_indices[0] = target_idx;
            topk_scores[0] = score;

            // Re-heapify
            int parent_idx = 0;
            while (true)
            {
                /* If the parent has no children, then we're done */
                const int left_child_idx = 2 * parent_idx + 1;
                if (left_child_idx >= k)
                {
                    break;
                }

                /* Find the node with the smallest score among the parent and its children */
                int smallest_idx = parent_idx;

                if (topk_scores[left_child_idx] < topk_scores[smallest_idx] ||
                    (topk_scores[left_child_idx] == topk_scores[smallest_idx] &&
                     topk_indices[left_child_idx] > topk_indices[smallest_idx]))
                {
                    smallest_idx = left_child_idx;
                }

                const int right_child_idx = left_child_idx + 1;
                if ((right_child_idx < k) && (topk_scores[right_child_idx] < topk_scores[smallest_idx] ||
                                              (topk_scores[right_child_idx] == topk_scores[smallest_idx] &&
                                               topk_indices[right_child_idx] > topk_indices[smallest_idx])))
                {
                    smallest_idx = right_child_idx;
                }

                /* If the parent is the smallest node, then we're done */
                if (smallest_idx == parent_idx)
                {
                    break;
                }

                /* Otherwise, swap the parent and the smallest node */
                const int tmp_idx = topk_indices[parent_idx];
                const double tmp_score = topk_scores[parent_idx];
                topk_indices[parent_idx] = topk_indices[smallest_idx];
                topk_scores[parent_idx] = topk_scores[smallest_idx];
                topk_indices[smallest_idx] = tmp_idx;
                topk_scores[smallest_idx] = tmp_score;

                /* Go down to the smallest node */
                parent_idx = smallest_idx;
            }
        }
    }
}

static void topk_tanimoto_search_1024(const unsigned char *query_fp,
                                      const int k,
                                      const int num_targets,
                                      const unsigned char *target_fps,
                                      int *topk_indices,
                                      double *topk_scores)
{
    const int num_bytes = 1024 / 8;

    /* Initialize the binary heap */
    for (int i = 0; i < k; i++)
    {
        topk_indices[i] = -1;
        topk_scores[i] = -1.0;
    }

    /* Do the search */
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const double score = byte_tanimoto_1024(query_fp, target_fps + target_idx * num_bytes);

        /* If the score is greater than the smallest score in the heap */
        if (score > topk_scores[0])
        {
            /* Replace the root of the heap with the new node */
            topk_indices[0] = target_idx;
            topk_scores[0] = score;

            // Re-heapify
            int parent_idx = 0;
            while (true)
            {
                /* If the parent has no children, then we're done */
                const int left_child_idx = 2 * parent_idx + 1;
                if (left_child_idx >= k)
                {
                    break;
                }

                /* Find the node with the smallest score among the parent and its children */
                int smallest_idx = parent_idx;

                if (topk_scores[left_child_idx] < topk_scores[smallest_idx] ||
                    (topk_scores[left_child_idx] == topk_scores[smallest_idx] &&
                     topk_indices[left_child_idx] > topk_indices[smallest_idx]))
                {
                    smallest_idx = left_child_idx;
                }

                const int right_child_idx = left_child_idx + 1;
                if ((right_child_idx < k) && (topk_scores[right_child_idx] < topk_scores[smallest_idx] ||
                                              (topk_scores[right_child_idx] == topk_scores[smallest_idx] &&
                                               topk_indices[right_child_idx] > topk_indices[smallest_idx])))
                {
                    smallest_idx = right_child_idx;
                }

                /* If the parent is the smallest node, then we're done */
                if (smallest_idx == parent_idx)
                {
                    break;
                }

                /* Otherwise, swap the parent and the smallest node */
                const int tmp_idx = topk_indices[parent_idx];
                const double tmp_score = topk_scores[parent_idx];
                topk_indices[parent_idx] = topk_indices[smallest_idx];
                topk_scores[parent_idx] = topk_scores[smallest_idx];
                topk_indices[smallest_idx] = tmp_idx;
                topk_scores[smallest_idx] = tmp_score;

                /* Go down to the smallest node */
                parent_idx = smallest_idx;
            }
        }
    }
}

static void topk_tanimoto_search_2048(const unsigned char *query_fp,
                                      const int k,
                                      const int num_targets,
                                      const unsigned char *target_fps,
                                      int *topk_indices,
                                      double *topk_scores)
{
    const int num_bytes = 2048 / 8;

    /* Initialize the binary heap */
    for (int i = 0; i < k; i++)
    {
        topk_indices[i] = -1;
        topk_scores[i] = -1.0;
    }

    /* Do the search */
    for (int target_idx = 0; target_idx < num_targets; target_idx++)
    {
        const double score = byte_tanimoto_2048(query_fp, target_fps + target_idx * num_bytes);

        /* If the score is greater than the smallest score in the heap */
        if (score > topk_scores[0])
        {
            /* Replace the root of the heap with the new node */
            topk_indices[0] = target_idx;
            topk_scores[0] = score;

            // Re-heapify
            int parent_idx = 0;
            while (true)
            {
                /* If the parent has no children, then we're done */
                const int left_child_idx = 2 * parent_idx + 1;
                if (left_child_idx >= k)
                {
                    break;
                }

                /* Find the node with the smallest score among the parent and its children */
                int smallest_idx = parent_idx;

                if (topk_scores[left_child_idx] < topk_scores[smallest_idx] ||
                    (topk_scores[left_child_idx] == topk_scores[smallest_idx] &&
                     topk_indices[left_child_idx] > topk_indices[smallest_idx]))
                {
                    smallest_idx = left_child_idx;
                }

                const int right_child_idx = left_child_idx + 1;
                if ((right_child_idx < k) && (topk_scores[right_child_idx] < topk_scores[smallest_idx] ||
                                              (topk_scores[right_child_idx] == topk_scores[smallest_idx] &&
                                               topk_indices[right_child_idx] > topk_indices[smallest_idx])))
                {
                    smallest_idx = right_child_idx;
                }

                /* If the parent is the smallest node, then we're done */
                if (smallest_idx == parent_idx)
                {
                    break;
                }

                /* Otherwise, swap the parent and the smallest node */
                const int tmp_idx = topk_indices[parent_idx];
                const double tmp_score = topk_scores[parent_idx];
                topk_indices[parent_idx] = topk_indices[smallest_idx];
                topk_scores[parent_idx] = topk_scores[smallest_idx];
                topk_indices[smallest_idx] = tmp_idx;
                topk_scores[smallest_idx] = tmp_score;

                /* Go down to the smallest node */
                parent_idx = smallest_idx;
            }
        }
    }
}
