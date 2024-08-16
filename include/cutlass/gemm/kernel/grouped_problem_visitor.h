/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Base scheduler for grouped problems
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/fast_math.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerated type describing the type of scheduling to perform for the ProblemVisitor
enum class GroupScheduleMode {
  // Perform all scheduling on device
  kDeviceOnly,
  // Precompute on the host the full sequence of problems to access
  kHostPrecompute,

  mixedStreamK,
};

/// Visitor class to abstract away the algorithm for iterating over tiles
template <typename ProblemSizeHelper,
          typename ThreadblockShape_>
struct BaseGroupedProblemVisitor {
  using ThreadblockShape = ThreadblockShape_;

  struct ProblemInfo {
    static int32_t const kNoPrefetchEntry = -1;
    int32_t problem_idx;
    int32_t problem_start;

    CUTLASS_DEVICE
    ProblemInfo() : problem_idx(kNoPrefetchEntry), problem_start(kNoPrefetchEntry) {}

    CUTLASS_DEVICE
    ProblemInfo(int32_t problem_idx_, int32_t problem_start_) :
      problem_idx(problem_idx_), problem_start(problem_start_) {}
  };

  struct Params {
    cutlass::gemm::GemmCoord const *problem_sizes;
    int32_t                         problem_count;
    void const                     *workspace;
    int32_t                         tile_count;

    //
    // Methods
    //

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(): problem_sizes(nullptr), problem_count(0), workspace(nullptr), tile_count(0) { }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const *problem_sizes,
      int32_t                         problem_count,
      void const                     *workspace = nullptr,
      int32_t                         tile_count = 0
    ):
      problem_sizes(problem_sizes),
      problem_count(problem_count),
      workspace(workspace),
      tile_count(tile_count)
    {}

  };

  Params params;
  int32_t tile_idx;
  int32_t problem_tile_start;
  int32_t problem_idx;

  //
  // Methods
  //
  CUTLASS_DEVICE
  BaseGroupedProblemVisitor(
    Params const &params_,
    int32_t block_idx
  ):
  params(params_),
  tile_idx(block_idx),
  problem_tile_start(0),
  problem_idx(0)
  {}

  /// Get the grid shape
  CUTLASS_HOST_DEVICE
  static cutlass::gemm::GemmCoord grid_shape(const cutlass::gemm::GemmCoord& problem) {
    return ProblemSizeHelper::grid_shape(problem);
  }

  /// Gets the global tile index
  CUTLASS_HOST_DEVICE
  int32_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  CUTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return problem_idx;
  }

  CUTLASS_HOST_DEVICE
  int32_t threadblock_idx() const {
    return tile_idx - problem_tile_start;
  }

  CUTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size;
  }

  CUTLASS_HOST_DEVICE
  static void possibly_transpose_problem(cutlass::gemm::GemmCoord& problem) {
    ProblemSizeHelper::possibly_transpose_problem(problem);
  }

  /// Returns the problem size for the current problem
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord problem_size() const {
    GemmCoord problem = params.problem_sizes[problem_idx];
    ProblemSizeHelper::possibly_transpose_problem(problem);
    return problem;
  }

  CUTLASS_HOST_DEVICE
  static int32_t tile_count(const cutlass::gemm::GemmCoord& grid) {
    return ProblemSizeHelper::tile_count(grid);
  }

  static int32_t group_tile_count(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr, int32_t problem_count) {
    int32_t total_tiles = 0;
    for (int32_t i = 0; i < problem_count; ++i) {
      auto problem = host_problem_sizes_ptr[i];
      possibly_transpose_problem(problem);
      auto grid = grid_shape(problem);
      total_tiles += tile_count(grid);
    }

    return total_tiles;
  }

  static size_t get_workspace_size_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {return 0;}
  static void host_precompute_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              //
                              int num_sms,
                              int sm_occupancy,
                              GemmCoord thread_block_shape,
                              //
                              void* host_workspace_ptr) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ProblemSizeHelper,
  typename ThreadblockShape,
  GroupScheduleMode GroupScheduleMode_,
  int PrefetchTileCount,
  int ThreadCount
>
struct GroupedProblemVisitor;

/////////////////////////////////////////////////////////////////////////////////////////////////
// ProblemVisitor that performs all scheduling on device
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::kDeviceOnly,
                             PrefetchTileCount,
                             ThreadCount>: public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  static int const kThreadCount = ThreadCount;
  static int const kRequiresPrecomputation = 0;
  static int const kThreadsPerWarp = 32;

  struct SharedStorage {};

  // Final tile of the problem loaded by this thread. Each thread will hold
  // a separate value.
  int32_t problem_ending_tile;

  SharedStorage &shared_storage;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  problem_ending_tile(0),
  shared_storage(shared_storage_)
  {
    this->problem_idx = -1 * kThreadsPerWarp;
    this->problem_tile_start = 0;

    if ((block_idx == 0 || block_idx == 1) && threadIdx.x == 0) {
      printf("tile_count: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, problem_ending_tile: %d\n", this->params.tile_count, this->tile_idx, this->problem_tile_start, this->problem_idx, this->problem_ending_tile);
    }

  }

  CUTLASS_DEVICE
  bool next_tile() {
    // Check whether the tile to compute is within the range of the current problem.
    int32_t problem_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, this->problem_idx % kThreadsPerWarp);
    if (this->tile_idx < problem_tile_end) {
      return true;
    }

    // Check whether the tile to compute is within the current group of problems fetched by the warp.
    // The last tile for this group is the final tile of the problem held by the final thread in the warp.
    int32_t group_tile_end = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp-1);

    // Keep the starting problem for this group in `problem_idx`. This is done to reduce
    // register pressure. The starting problem for this group is simply the first problem
    // in the group most recently fetched by the warp.
    int32_t &group_problem_start = this->problem_idx;
    group_problem_start = (this->problem_idx / kThreadsPerWarp) * kThreadsPerWarp;

    // Keep the starting tile for this group in `problem_tile_start`. This is done to reduce
    // register pressure.
    int32_t &group_tile_start = this->problem_tile_start;

    // Each thread in the warp processes a separate problem to advance until
    // reaching a problem whose starting tile is less less than tile_idx.
    while (group_tile_end <= this->tile_idx) {
      group_problem_start += kThreadsPerWarp;
      if (group_problem_start > this->params.problem_count) {
        return false;
      }

      // Since `group_tile_start` is a reference to `this->problem_tile_start`, this
      // also sets `this->problem_tile_start`. The fact that `this->problem_tile_start`
      // is also set here is used later in `next_tile`.
      group_tile_start = group_tile_end;

      int lane_idx = threadIdx.x % kThreadsPerWarp;
      int32_t lane_problem = group_problem_start + lane_idx;

      // Compute the number of tiles in the problem assigned to each thread.
      problem_ending_tile = 0;
      if (lane_problem < this->params.problem_count) {
        cutlass::gemm::GemmCoord problem = this->params.problem_sizes[lane_problem];
        this->possibly_transpose_problem(problem);
        cutlass::gemm::GemmCoord grid = this->grid_shape(problem);
        problem_ending_tile = this->tile_count(grid);
      }

      // Compute a warp-wide inclusive prefix sum to compute the ending tile index of
      // each thread's problem.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < kThreadsPerWarp; i <<= 1) {
        int32_t val = __shfl_up_sync(0xffffffff, problem_ending_tile, i);
        if (lane_idx >= i) {
          problem_ending_tile += val;
        }
      }

      // The total tile count for this group is now in the final position of the prefix sum
      int32_t tiles_in_group = __shfl_sync(0xffffffff, problem_ending_tile, kThreadsPerWarp-1);

      problem_ending_tile += group_tile_start;
      group_tile_end += tiles_in_group;
    }

    // The next problem to process is the first one that does not have ending tile position
    // that is greater than or equal to tile index.
    int32_t problem_idx_in_group =
        __popc(__ballot_sync(0xffffffff, problem_ending_tile <= this->tile_idx));

    this->problem_idx = group_problem_start + problem_idx_in_group;

    // The starting tile for this problem is the ending tile of the previous problem. In cases
    // where `problem_idx_in_group` is the first problem in the group, we do not need to reset
    // `problem_tile_start`, because it is set to the previous group's ending tile in the while
    // loop above.
    if (problem_idx_in_group > 0) {
      this->problem_tile_start = __shfl_sync(0xffffffff, problem_ending_tile, problem_idx_in_group - 1);
    }

    if ((blockIdx.x == 0) && threadIdx.x == 0) {
      // printf("[NEXT] tile_count: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, problem_ending_tile: %d , problem_tile_end: %d\n", this->params.tile_count, this->tile_idx, this->problem_tile_start, this->problem_idx, this->problem_ending_tile, problem_tile_end);

      printf("[Next] tiled_idx: %d, problem_tile_start: %d, problem_tile_end: %d, group_tile_start: %d, group_tile_end: %d, problem_idx: %d, problem_index_in_group: %d\n", this->tile_idx, this->problem_tile_start, problem_tile_end, group_tile_start, group_tile_end, this->problem_idx, problem_idx_in_group);
    }

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    return 0;
  }

  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Precomputes schedule on host and prefetches into shared memory
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::kHostPrecompute,
                             PrefetchTileCount,
                             ThreadCount> : public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  static_assert(PrefetchTileCount > 0,
                "GroupedProblemVisitor with GroupScheduleMode `kHostPrecompute` currently requires prefetching to shared memory");

  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  using ProblemInfo = typename Base::ProblemInfo;
  static int const kRequiresPrecomputation = 1;

  static int const kPrefetchTileCount = PrefetchTileCount;
  static int const kThreadCount = ThreadCount;

  struct SharedStorage {
    // Sequence of problem IDs and starting tiles to compute
    cutlass::Array<ProblemInfo, kPrefetchTileCount> prefetched_problems;
  };

  int32_t tiles_computed;
  int32_t iterations_per_block;
  int32_t block_load_start;
  SharedStorage &shared_storage;
  ProblemInfo const *problem_info_ptr;

  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx),
  tiles_computed(0),
  shared_storage(shared_storage_),
  problem_info_ptr(reinterpret_cast<ProblemInfo const*>(params_.workspace))
  {
    iterations_per_block = (params_.tile_count - 1 + gridDim.x) / gridDim.x;
    block_load_start = iterations_per_block * block_idx;

    if ((block_idx == 0 || block_idx == 1 || block_idx==127) && threadIdx.x == 0) {
      printf("tile_count: %d, tiles_computed: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, iterations_per_block: %d, block_load_start: %d, kPrefetchTileCount: %d, kThreadCount: %d\n", params_.tile_count, tiles_computed, this->tile_idx, this->problem_tile_start, this->problem_idx, iterations_per_block, block_load_start, kPrefetchTileCount, kThreadCount);
    }

    // Start prefetching the first set of tiles to compute
    prefetch_tiles();
  }

  CUTLASS_DEVICE
  bool next_tile() {
    if (this->tile_idx >= this->params.tile_count) {
      return false;
    }

    int32_t prefetch_idx = (tiles_computed % kPrefetchTileCount);
    if (prefetch_idx == 0) {
      // Ensure all previous stores to shared memory have been completed
      __syncthreads();
    }

    auto problem_info = shared_storage.prefetched_problems[prefetch_idx];
    ++tiles_computed;

    if ((tiles_computed % kPrefetchTileCount) == 0) {
      // Begin prefetching next set of tiles. Synchronize first to ensure that
      // we don't overwrite the current buffer while someone else is using it.
      __syncthreads();
      prefetch_tiles();
    }

    this->problem_idx = problem_info.problem_idx;
    this->problem_tile_start = problem_info.problem_start;

    if ((blockIdx.x == 0) && threadIdx.x == 0) {
      // printf("[NEXT] tile_count: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, problem_ending_tile: %d , problem_tile_end: %d\n", this->params.tile_count, this->tile_idx, this->problem_tile_start, this->problem_idx, this->problem_ending_tile, problem_tile_end);
      printf("[Next] tiled_idx: %d, prefetch_idx: %d, tiles_computed: %d, problem_idx: %d, problem_tile_start: %d\n", this->tile_idx, prefetch_idx, tiles_computed, this->problem_idx, this->problem_tile_start); 
    }

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = ((total_tiles - 1 + block_count) / block_count);
    return sizeof(ProblemInfo) * entries_per_block * block_count;
  }
#if !defined(__CUDACC_RTC__)
  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) {
    ProblemInfo* host_problem_info_ptr = reinterpret_cast<ProblemInfo*>(host_workspace_ptr);
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = (total_tiles - 1 + block_count) / block_count;

    int tile = 0;
    int start_tile = 0;
    for (int p_idx = 0; p_idx < problem_count; ++p_idx) {
      auto problem = host_problem_sizes_ptr[p_idx];
      Base::possibly_transpose_problem(problem);
      auto grid = Base::grid_shape(problem);
      int tiles = Base::tile_count(grid);
      ProblemInfo problem_info(p_idx, start_tile);
      for (int i = 0; i < tiles; ++i, ++tile) {
        host_problem_info_ptr[(entries_per_block * (tile % block_count)) + (tile / block_count)] = problem_info;
      }
      start_tile += tiles;
    }
  }
#endif
private:
  CUTLASS_DEVICE
  void prefetch_tiles() {
    CUTLASS_PRAGMA_UNROLL
    for (int32_t i = 0; i < kPrefetchTileCount; i += kThreadCount) {
      int32_t offset = threadIdx.x + i;
      if (offset < kPrefetchTileCount && (tiles_computed + offset < iterations_per_block)) {
        shared_storage.prefetched_problems[offset] = problem_info_ptr[block_load_start + tiles_computed + offset];
      }
    }
  }
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// mixed streamK
//
template <typename ProblemSizeHelper,
          typename ThreadblockShape,
          int PrefetchTileCount,
          int ThreadCount>
struct GroupedProblemVisitor<ProblemSizeHelper,
                             ThreadblockShape,
                             GroupScheduleMode::mixedStreamK,
                             PrefetchTileCount,
                             ThreadCount> : public BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape> {
  static_assert(PrefetchTileCount > 0,
                "mixedStreamK currently requires prefetching to shared memory");

  using Base = BaseGroupedProblemVisitor<ProblemSizeHelper, ThreadblockShape>;
  using Params = typename Base::Params;
  using ProblemInfo = typename Base::ProblemInfo;

  static int const kRequiresPrecomputation = 2;

  static int const kPrefetchTileCount = PrefetchTileCount;
  static int const kThreadCount = ThreadCount;

  // customized with sk info
  struct skInfo {
    int sk_regions;
    int dp_blocks;
    int dp_tiles;
    int sk_blocks;
    int sk_tiles;
    int sk_waves;

    CUTLASS_DEVICE
    skInfo(
    int sk_regions,
    int dp_blocks,
    int dp_tiles,
    int sk_blocks,
    int sk_tiles,
    int sk_waves
    ) : sk_regions(sk_regions)
        ,dp_blocks(dp_blocks)
        ,dp_tiles(dp_tiles)
        ,sk_blocks(sk_blocks)
        ,sk_tiles(sk_tiles)
        ,sk_waves(sk_waves)
         {}
  };

  struct SharedStorage {
    // Sequence of problem IDs and starting tiles to compute
    cutlass::Array<ProblemInfo, kPrefetchTileCount> prefetched_problems;
  };

  int32_t tiles_computed;
  int32_t iterations_per_block;
  int32_t block_load_start;
  SharedStorage &shared_storage;
  ProblemInfo const *problem_info_ptr;

  //
  //
  //
  int dp_start_tile_idx{0};
  int dp_start_block_idx{0};
  int32_t block_idx;
  bool is_sk;


  //
  // heuristic
  //

  //
  static int const kMinItersPerSkBlock = 2;

  /// Height in CTAs of a grid rasterization cohort
  // static int const kCohortCtasM = 8;
  // /// Width in CTAs of a grid rasterization cohort
  // static int const kCohortCtasN = 4;
  // static int const kCtasPerCohort = kCohortCtasN * kCohortCtasM;


  //
  // Methods
  //
  CUTLASS_DEVICE
  GroupedProblemVisitor(
    Params const &params_,
    SharedStorage &shared_storage_,
    int32_t block_idx
  ): Base(params_, block_idx/*tile_idx*/),
  tiles_computed(0),
  shared_storage(shared_storage_),
  //problem_info_ptr(reinterpret_cast<ProblemInfo const*>(params_.workspace)),
  problem_info_ptr(nullptr),
  block_idx(block_idx)
  {
    //
    // sk related
    //
    skInfo const *sk_info_ptr = reinterpret_cast<skInfo const*>(params_.workspace);
    auto p = *sk_info_ptr;

    int dp_start_block_idx = p.sk_blocks * p.sk_waves;
    int dp_start_tile_idx = p.sk_tiles;

    if (block_idx == 0 && threadIdx.x == 0) {
      printf("[SK]: dp_start_block_idx: %d, dp_start_tile_idx: %d\n", dp_start_block_idx, dp_start_tile_idx);
    }
    this->dp_start_tile_idx = dp_start_tile_idx;
    this->dp_start_block_idx = dp_start_block_idx;
    if (dp_start_block_idx > 0)
      is_sk = true;

    // 
    // advance to problem info ptr
    //
    char const* char_ptr = reinterpret_cast<char const*>(params_.workspace);
    char_ptr += sizeof(skInfo);
    this->problem_info_ptr = reinterpret_cast<ProblemInfo const*>(char_ptr);

    // 
    // group gemm
    //
    iterations_per_block = (params_.tile_count - 1 + gridDim.x) / gridDim.x;
    block_load_start = iterations_per_block * block_idx;

    if ((block_idx == 0 || block_idx == 1 || block_idx==127) && threadIdx.x == 0) {

      printf("[ProblemVisitor]: tile_count: %d, tiles_computed: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, iterations_per_block: %d, block_load_start: %d, kPrefetchTileCount: %d, kThreadCount: %d\n", params_.tile_count, tiles_computed, this->tile_idx, this->problem_tile_start, this->problem_idx, iterations_per_block, block_load_start, kPrefetchTileCount, kThreadCount);

    }

    // Start prefetching the first set of tiles to compute
    prefetch_tiles();
  }


  CUTLASS_DEVICE
  bool next_tile() {
    if (this->tile_idx >= this->params.tile_count) {
      return false;
    }

    int32_t prefetch_idx = (tiles_computed % kPrefetchTileCount);
    if (prefetch_idx == 0) {
      // Ensure all previous stores to shared memory have been completed
      __syncthreads();
    }

    auto problem_info = shared_storage.prefetched_problems[prefetch_idx];
    ++tiles_computed;

    if ((tiles_computed % kPrefetchTileCount) == 0) {
      // Begin prefetching next set of tiles. Synchronize first to ensure that
      // we don't overwrite the current buffer while someone else is using it.
      __syncthreads();
      prefetch_tiles();
    }

    this->problem_idx = problem_info.problem_idx;
    this->problem_tile_start = problem_info.problem_start;

    if ((blockIdx.x == 0) && threadIdx.x == 0) {
      // printf("[NEXT] tile_count: %d, tile_idx: %d, problem_tile_start: %d, problem_idx: %d, problem_ending_tile: %d , problem_tile_end: %d\n", this->params.tile_count, this->tile_idx, this->problem_tile_start, this->problem_idx, this->problem_ending_tile, problem_tile_end);
      printf("[Next] tiled_idx: %d, prefetch_idx: %d, tiles_computed: %d, problem_idx: %d, problem_tile_start: %d\n", this->tile_idx, prefetch_idx, tiles_computed, this->problem_idx, this->problem_tile_start); 
    }

    return true;
  }

  static size_t get_workspace_size(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) { assert(false); return 0; }
  static void host_precompute(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              void* host_workspace_ptr) { assert(false);}

  static skInfo plan_sk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int num_sms,
                              int sm_occupancy,
                              GemmCoord thread_block_shape,
                              bool verbose
                              ) {
    // total tiles for all gemm
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);

    //
    // Determine dispatch composition of DP-tiles and SK-blocks
    //
    auto first_problem = host_problem_sizes_ptr[0];
    int iters_per_tile = (first_problem.k() + thread_block_shape.k() - 1) / thread_block_shape.k();
    int waves = (total_tiles + num_sms - 1) / num_sms;
    float dp_efficiency = float(total_tiles) / float(waves * num_sms);

    if (verbose)
      std::cout << "total_tiles: " << total_tiles << ", iters_per_tile: " << iters_per_tile << ", waves: " << waves << ", dp_efficiency: " << dp_efficiency << std::endl;

    // Start with a DP-only configuration
    int dp_tiles = total_tiles;     // Number of data-parallel tiles

    //
    // get_blocks
    //
    int full_waves = total_tiles / num_sms;
    int full_waves_tiles = full_waves * num_sms;
    int partial_wave_tiles = total_tiles - full_waves_tiles;
    if (partial_wave_tiles == 0) {
      // TODO Perfect quantization
      assert(false);
    }
    if (full_waves < sm_occupancy) {
      // cornor case: We're less than full GPU occupancy
      assert(false);
    }
    if (sm_occupancy > 1) {
      // ???
      assert(false);
    }

    dp_tiles = full_waves_tiles - num_sms;  // this gives 1 wave to sk
    int max_sk_occupancy = sm_occupancy - ((full_waves - 1) % sm_occupancy);

    //
    // get_sk_blocks
    //
    int sk_tiles = partial_wave_tiles + num_sms;
    int sk_iters = sk_tiles * iters_per_tile;

    int dp_equiv_waves = (sk_tiles + num_sms - 1) / num_sms;
    int dp_equiv_iters = iters_per_tile * dp_equiv_waves;

    int min_sk_blocks = fast_min(num_sms, sk_tiles + 1);
    int max_sk_blocks = fast_min(num_sms * max_sk_occupancy, sk_iters / kMinItersPerSkBlock);

    if (verbose)
      std::cout << "sk_tiles: " << sk_tiles << ", sk_iters: " << sk_iters << ", max_sk_occupancy: " << max_sk_occupancy << ", dp_equiv_waves: " << dp_equiv_waves << ", dp_equiv_iters: " << dp_equiv_iters << ", min_sk_blocks: " << min_sk_blocks << ", max_sk_blocks: " << max_sk_blocks << std::endl;

    // Number of thread blocks to produce the remaining SK tiles
    int sk_blocks = 0;              
    int savings_iters = INT_MIN;
    {
      // heuristic for picking sk_blocks
      for (int trial_sk_blocks = min_sk_blocks; trial_sk_blocks <= max_sk_blocks; ++trial_sk_blocks) {
        int sk_waves = (trial_sk_blocks + num_sms - 1) / num_sms;

        int max_sk_iters_per_block = (sk_iters + trial_sk_blocks - 1) / trial_sk_blocks;
        int sk_iter_equiv = max_sk_iters_per_block * sk_waves;

        int num_peers = ((trial_sk_blocks + sk_tiles - 1) / sk_tiles) + 1;        // add one for alignment skew

        float iter_cost = 0.02f * float(num_peers) * float(sk_iter_equiv);

        if (trial_sk_blocks % sk_tiles == 0)
        {
          // aligned
          num_peers = (trial_sk_blocks / sk_tiles);

          iter_cost = 0.0f;
        }

        float peer_cost = 2.0f * float(num_peers);

        float base_cost = 2.0f * float(sk_waves);

        int fixup_iter_equiv = int(base_cost + iter_cost + peer_cost);

        int trial_savings_iters = dp_equiv_iters - sk_iter_equiv - fixup_iter_equiv;

        if (trial_savings_iters >= savings_iters) {
            savings_iters = trial_savings_iters;
            sk_blocks = trial_sk_blocks;
        }
      }
    }
    if (savings_iters < 0) {
      // not profitable; TODO apply a dp setting
      assert(false);
    }


    //
    // post-process sk region
    //
    int sk_regions = 1;
    //int sk_iters_per_normal_block;
    // int reduction_blocks = 0;
    // bool remap_block_indices = false;
    int sk_waves = -1;
    assert(sk_blocks > 0);
    {
      sk_waves = (sk_blocks + num_sms - 1) / num_sms;
      int sk_iters = sk_tiles * iters_per_tile;
      sk_blocks = fast_min(sk_blocks, sk_iters);
      //sk_iters_per_normal_block = sk_iters / sk_blocks;
      // int extra_sk_iters = sk_iters - (sk_iters_per_normal_block * sk_blocks);
      // int sk_big_blocks = extra_sk_iters;
      if ((sk_blocks > sk_tiles) && (sk_blocks % sk_tiles == 0)) {
        // // Split-K decomposition
        // sk_regions = sk_tiles;
        assert(false);
      }

      // int sk_blocks_per_region = sk_blocks / sk_regions;
      // int sk_big_blocks_per_region = sk_big_blocks / sk_regions;
      // int sk_iters_per_region = sk_iters / sk_regions;
    }


    //
    // Compute DP blocks
    // TODO: dont think we need cohort raster in grouped GEMM
    //

    int dp_blocks = dp_tiles;
    // GemmCoord tiled_shape(
    //   (first_problem.m() + thread_block_shape.m() - 1) / thread_block_shape.m(),
    //   (first_problem.n() + thread_block_shape.n() - 1) / thread_block_shape.n(),
    //   1);
    // cutlass::gemm::GemmCoord tiled_cohort_shape(
    //     (tiled_shape.m() + kCohortCtasM - 1) / kCohortCtasM,
    //     (tiled_shape.n() + kCohortCtasN - 1) / kCohortCtasN,
    //     tiled_shape.k());
    // int cohort_blocks = (tiled_cohort_shape.m() * tiled_cohort_shape.n()) * kCtasPerCohort;
    // float cohort_efficiency = float(dp_blocks) / float(cohort_blocks);


    //
    auto sk_info = skInfo(
         sk_regions,
         dp_blocks,
         dp_tiles,
         sk_blocks,
         sk_tiles,
         sk_waves
    );
    if (verbose)
      std::cout << "sk regions: " << sk_info.sk_regions << ", dp blocks: " << sk_info.dp_blocks << ", dp tiles: " << sk_info.dp_tiles << ", sk blocks: " << sk_info.sk_blocks << ", sk tiles: " << sk_info.sk_tiles << ", sk_waves: " << sk_info.sk_waves <<  std::endl;
    return sk_info;
  }

  static size_t get_workspace_size_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                                   int32_t problem_count,
                                   int32_t block_count) {
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = ((total_tiles - 1 + block_count) / block_count);
    return sizeof(skInfo) + sizeof(ProblemInfo) * entries_per_block * block_count;
  }
  static void host_precompute_streamk(const cutlass::gemm::GemmCoord* host_problem_sizes_ptr,
                              int32_t problem_count,
                              int32_t block_count,
                              //
                              int num_sms,
                              int sm_occupancy,
                              GemmCoord thread_block_shape,
                              //
                              void* host_workspace_ptr) {
    std::cout << "[Precompute] " << std::endl;
    auto sk_info = plan_sk(host_problem_sizes_ptr, 
                      problem_count,
                      num_sms,
                      sm_occupancy,
                      thread_block_shape,
                      true);

    //
    // save to work space
    //
    skInfo *sk_info_ptr = reinterpret_cast<skInfo*>(host_workspace_ptr);
    *sk_info_ptr = sk_info;

    //
    // advance to problem info ptr
    //
    char *char_ptr = reinterpret_cast<char*>(host_workspace_ptr);
    char_ptr += sizeof(skInfo);
    ProblemInfo* host_problem_info_ptr = reinterpret_cast<ProblemInfo*>(char_ptr);

    // 
    // based on sk info, plan the rest dp blocks  
    //
    int32_t total_tiles = Base::group_tile_count(host_problem_sizes_ptr, problem_count);
    int32_t entries_per_block = (total_tiles - 1 + block_count) / block_count;
    // assert(sk_info.dp_tiles % block_count == 0);  // <- perfect quantization!!!
    // int32_t entries_per_block = sk_info.dp_tiles/block_count;

    int tile = 0;
    int start_tile = 0;
    for (int p_idx = 0; p_idx < problem_count; ++p_idx) {
      auto problem = host_problem_sizes_ptr[p_idx];
      Base::possibly_transpose_problem(problem);
      auto grid = Base::grid_shape(problem);
      int tiles = Base::tile_count(grid);
      ProblemInfo problem_info(p_idx, start_tile);
      for (int i = 0; i < tiles; ++i, ++tile) {

        //
        // for [0 - entries_per_block], it's the work belongs to block 0
        //
        // for problems 0, if it has 128 tiles (assume 128 sms), each sm takes one tile, 
        // so its tiles are distributed across sms
        //

        //if (tile < sk_info.sk_tiles) {
        //  ; // todo sk 
        //} else {
        //  int dp_tile = tile - sk_info.sk_tiles;
        //  host_problem_info_ptr[(entries_per_block * (dp_tile % block_count)) + (dp_tile / block_count)] = problem_info;
        //}

        host_problem_info_ptr[(entries_per_block * (tile % block_count)) + (tile / block_count)] = problem_info;
      }
      start_tile += tiles;
    }

    std::cout << "[Precompute Done]" << std::endl;
  }
private:
  CUTLASS_DEVICE
  void prefetch_tiles() {

    // 
    // each thread has a problem visitor
    // however the total amount of work is: (total_tiles - 1 + block_count) / block_count
    //

    CUTLASS_PRAGMA_UNROLL
    for (int32_t i = 0; i < kPrefetchTileCount; i += kThreadCount) {
      int32_t offset = threadIdx.x + i;
      if (offset < kPrefetchTileCount && (tiles_computed + offset < iterations_per_block)) {
        shared_storage.prefetched_problems[offset] = problem_info_ptr[block_load_start + tiles_computed + offset];
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
