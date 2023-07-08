// Copyright 2022 The ABCBoost Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <math.h>
#include <algorithm>  // std::sort, std::min, std::max
#include <iterator>
#include <numeric>  // std::iota
#include <queue>    // std::priority_queue
#include <set>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>

#include "config.h"
#ifdef OMP
#include "omp.h"
#else
#include "dummy_omp.h"
#endif
#include "tree.h"
#include "utils.h"


namespace ABCBoost {

inline HistBin csw_plus(const HistBin& a, const HistBin& b){
  return HistBin(a.count + b.count,a.sum + b.sum,a.weight + b.weight);
}

#ifndef OS_WIN
#pragma omp declare reduction(vec_csw_plus :  std::vector<HistBin>: \
  std::transform( \
    omp_out.begin(), omp_out.end(), \
    omp_in.begin(), omp_out.begin(), csw_plus)) \
  initializer(omp_priv = omp_orig)

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
  std::transform( \
    omp_out.begin(), omp_out.end(), \
    omp_in.begin(), omp_out.begin(), std::plus<double>())) \
  initializer(omp_priv = omp_orig)

#pragma omp declare reduction(vec_int_plus: std::vector<int> : \
  std::transform( \
    omp_out.begin(), omp_out.end(), \
    omp_in.begin(), omp_out.begin(), std::plus<int>())) \
  initializer(omp_priv = omp_orig)
#endif
/**
 * Constructor.
 * @param[in] data: Dataset to train on
 *            config: Configuration
 */
Tree::Tree(Data *data, Config *config) {
  this->config = config;
  this->data = data;
  n_leaves = config->tree_max_n_leaves;
  n_threads = config->n_threads;
  nodes.resize(2 * n_leaves - 1);
  is_weighted = config->model_use_logit;
}

Tree::~Tree() {
  std::vector<short>().swap(leaf_ids);
  std::vector<TreeNode>().swap(nodes);
}

Tree::TreeNode::TreeNode() {
  is_leaf = true;
  allow_build_subtree = true;
  is_random_node = false;
  has_retrain = false;
  start = end = -1;
  idx = left = right = parent = -1;
  split_fi = split_v = -1;
  gain = predict_v = -1;
  gains.clear();
}


inline void Tree::alignHessianResidual(const uint start,const uint end){
  const auto* H = hessian;
  const auto* R = residual;
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static),
    config->use_omp == 1 && end - start > 1024,
    for(uint i = start;i < end;++i){
      auto id = ids[i];
      H_tmp[i] = H[id];
      R_tmp[i] = R[id];
    }
  )
}

inline void Tree::alignHessianResidual(const uint start,const uint end, std::vector<uint>& ids){
  const auto* H = hessian;
  const auto* R = residual;
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static),
    config->use_omp == 1 && end - start > 1024,
    for(uint i = start;i < end;++i){
      auto id = ids[i];
      H_tmp[i] = H[id];
      R_tmp[i] = R[id];
    }
  )
}

inline void Tree::alignHessianResidual(const uint start,const uint end, double* hessian, double* residual, std::vector<uint>& ids){
  const auto* H = hessian;
  const auto* R = residual;
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static),
    config->use_omp == 1 && end - start > 1024,
    for(uint i = start;i < end;++i){
      auto id = ids[i];
      H_tmp[i] = H[id];
      R_tmp[i] = R[id];
    }
  )
}


inline void Tree::initUnobserved(const uint start,const uint end,int &c_unobserved, double& r_unobserved, double& h_unobserved){
  const auto* H = hessian;
  const auto* R = residual;
  double r = r_unobserved;
  double h = h_unobserved;
  int c = c_unobserved;
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static) reduction(+: r, h, c),
    config->use_omp == true && (end - start > 1024),
    for (int i = start; i < end; ++i) {
      auto id = ids[i];
      r += R[id];
      h += H[id];
      c++;
    }
  )
  r_unobserved = r;
  h_unobserved = h;
  c_unobserved = c;
}

inline void Tree::initUnobserved(const uint start,const uint end,int &c_unobserved, double& r_unobserved, double& h_unobserved, std::vector<uint>& ids){
  const auto* H = hessian;
  const auto* R = residual;
  double r = r_unobserved;
  double h = h_unobserved;
  int c = c_unobserved;
  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static) reduction(+: r, h, c),
    config->use_omp == true && (end - start > 1024),
    for (int i = start; i < end; ++i) {
      auto id = ids[i];
      r += R[id];
      h += H[id];
      c++;
    }
  )
  r_unobserved = r;
  h_unobserved = h;
  c_unobserved = c;
}

/**
 * Calculate bin_counts and bin_sums for all features at a node.
 * @param[in] x: Node id
 *            sib: Sibling id
 * @post bin_counts[x] and bin_sums[x] are populated.
 */
void Tree::binSort(int x, int sib) {
  const auto* H = hessian;
  const auto* R = residual;
  uint start = nodes[x].start;
  uint end = nodes[x].end;
  uint fsz = fids->size();

  if (sib == -1) {
    if(!(start == 0 && end == data->n_data)){
      alignHessianResidual(start,end);
    }

    double r_unobserved = 0.0;
    double h_unobserved = 0.0;
    int c_unobserved = 0;
    initUnobserved(start,end,c_unobserved,r_unobserved,h_unobserved);
    
    setInLeaf<true>(start,end);

    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == 1,
    for (int j = 0; j < fsz; ++j) {
      int fid = (data->valid_fi)[(*fids)[j]];
      auto &b_csw = (*hist)[x][fid];
      if(data->auxDataWidth[fid] == 0){
        std::vector<data_quantized_t> &fv = (data->Xv)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = fv[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[i];
              b_csw[bin_id].weight += is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = fv[ids[i]];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R_tmp[i];
              b_csw[bin_id].weight += is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = fv[i];
              auto id = fi[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[id];
              b_csw[bin_id].weight += is_weighted ? H[id] : 1;
          
              b_csw[j_unobserved].sum -= R[id];
              b_csw[j_unobserved].count -= 1;
              b_csw[j_unobserved].weight -= is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count += c_unobserved;
          b_csw[j_unobserved].sum += r_unobserved;
          b_csw[j_unobserved].weight += is_weighted ? h_unobserved : c_unobserved;
        }
      }else{
        std::vector<uint8_t> &fv = (data->auxData)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[i];
              b_csw[bin_id].weight += is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[ids[i] >> 1] >> ((ids[i] & 1) << 2)) & 15;
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R_tmp[i];
              b_csw[bin_id].weight += is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              auto id = fi[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[id];
              b_csw[bin_id].weight += is_weighted ? H[id] : 1;
          
              b_csw[j_unobserved].sum -= R[id];
              b_csw[j_unobserved].count -= 1;
              b_csw[j_unobserved].weight -= is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count += c_unobserved;
          b_csw[j_unobserved].sum += r_unobserved;
          b_csw[j_unobserved].weight += is_weighted ? h_unobserved : c_unobserved;
        }
      }
    }
    )

    setInLeaf<false>(start,end);

  } else {
    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == true,
      for (int j = 0; j < fsz; ++j) {
        uint fid = (data->valid_fi)[(*fids)[j]];
        std::vector<HistBin> &b_csw = (*hist)[x][fid];
        int parent = nodes[x].parent;
        std::vector<HistBin> &pb_csw =
            (*hist)[parent][fid];
        std::vector<HistBin> &sb_csw = (*hist)[sib][fid];
        for (int k = 0; k < b_csw.size(); ++k) {
          b_csw[k].count = pb_csw[k].count - sb_csw[k].count;
          b_csw[k].sum = pb_csw[k].sum - sb_csw[k].sum;
          b_csw[k].weight =
              std::max((hist_t).0, pb_csw[k].weight - sb_csw[k].weight);
        }
      }
    )
  }
}

void Tree::unlearnBinSort(int x, int sib, uint start, uint end, std::vector<uint>& ids) {
  if (end - start <= 0) return;
  const auto* H = hessian;
  const auto* R = residual;
  uint fsz = fids->size();

  if (sib == -1) {
    if(!(start == 0 && end == data->n_data)){
      alignHessianResidual(start,end, hessian, residual, ids);
    }

    double r_unobserved = 0.0;
    double h_unobserved = 0.0;
    int c_unobserved = 0;
    initUnobserved(start,end,c_unobserved,r_unobserved,h_unobserved,ids);

    setInLeaf<true>(start,end,ids);

    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == 1,
    for (int j = 0; j < fsz; ++j) {
      int fid = (data->valid_fi)[(*fids)[j]];
      auto &b_csw = (*hist)[x][fid];
      if(data->auxDataWidth[fid] == 0){
        std::vector<data_quantized_t> &fv = (data->Xv)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = fv[i];
              b_csw[bin_id].count -= 1;
              b_csw[bin_id].sum -= R[i];
              b_csw[bin_id].weight -= is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = fv[ids[i]];
              b_csw[bin_id].count -= 1;
              b_csw[bin_id].sum -= R_tmp[i];
              b_csw[bin_id].weight -= is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = fv[i];
              auto id = fi[i];
              b_csw[bin_id].count -= 1;
              b_csw[bin_id].sum -= R[id];
              b_csw[bin_id].weight -= is_weighted ? H[id] : 1;

              b_csw[j_unobserved].sum += R[id];
              b_csw[j_unobserved].count += 1;
              b_csw[j_unobserved].weight += is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count -= c_unobserved;
          b_csw[j_unobserved].sum -= r_unobserved;
          b_csw[j_unobserved].weight -= is_weighted ? h_unobserved : c_unobserved;
        }
      }else{
        std::vector<uint8_t> &fv = (data->auxData)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              b_csw[bin_id].count -= 1;
              b_csw[bin_id].sum -= R[i];
              b_csw[bin_id].weight -= is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[ids[i] >> 1] >> ((ids[i] & 1) << 2)) & 15;
              b_csw[bin_id].count -= 1;
              b_csw[bin_id].sum -= R_tmp[i];
              b_csw[bin_id].weight -= is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              auto id = fi[i];
              b_csw[bin_id].count -= 1;
              b_csw[bin_id].sum -= R[id];
              b_csw[bin_id].weight -= is_weighted ? H[id] : 1;

              b_csw[j_unobserved].sum += R[id];
              b_csw[j_unobserved].count += 1;
              b_csw[j_unobserved].weight += is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count -= c_unobserved;
          b_csw[j_unobserved].sum -= r_unobserved;
          b_csw[j_unobserved].weight -= is_weighted ? h_unobserved : c_unobserved;
        }
      }
    }
    )

    setInLeaf<false>(start,end,ids);

  } else {
    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == true,
      for (int j = 0; j < fsz; ++j) {
        uint fid = (data->valid_fi)[(*fids)[j]];
        std::vector<HistBin> &b_csw = (*hist)[x][fid];
        int parent = nodes[x].parent;
        std::vector<HistBin> &pb_csw =
            (*hist)[parent][fid];
        std::vector<HistBin> &sb_csw = (*hist)[sib][fid];
        for (int k = 0; k < b_csw.size(); ++k) {
          b_csw[k].count = pb_csw[k].count - sb_csw[k].count;
          b_csw[k].sum = pb_csw[k].sum - sb_csw[k].sum;
          b_csw[k].weight =
              std::max((hist_t).0, pb_csw[k].weight - sb_csw[k].weight);
        }
      }
    )
  }
}

void Tree::tuneBinSort(int x, int sib, uint start, uint end, std::vector<uint>& ids) {
  if (end - start <= 0) return;
  const auto* H = hessian;
  const auto* R = residual;
  uint fsz = fids->size();

  if (sib == -1) {
    if(!(start == 0 && end == data->n_data)){
      alignHessianResidual(start,end, hessian, residual, ids);
    }

    double r_unobserved = 0.0;
    double h_unobserved = 0.0;
    int c_unobserved = 0;
    initUnobserved(start,end,c_unobserved,r_unobserved,h_unobserved,ids);

    setInLeaf<true>(start,end,ids);

    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == 1,
    for (int j = 0; j < fsz; ++j) {
      int fid = (data->valid_fi)[(*fids)[j]];
      auto &b_csw = (*hist)[x][fid];
      if(data->auxDataWidth[fid] == 0){
        std::vector<data_quantized_t> &fv = (data->Xv)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = fv[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[i];
              b_csw[bin_id].weight += is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = fv[ids[i]];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R_tmp[i];
              b_csw[bin_id].weight += is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = fv[i];
              auto id = fi[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[id];
              b_csw[bin_id].weight += is_weighted ? H[id] : 1;

              b_csw[j_unobserved].sum -= R[id];
              b_csw[j_unobserved].count -= 1;
              b_csw[j_unobserved].weight -= is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count += c_unobserved;
          b_csw[j_unobserved].sum += r_unobserved;
          b_csw[j_unobserved].weight += is_weighted ? h_unobserved : c_unobserved;
        }
      }else{
        std::vector<uint8_t> &fv = (data->auxData)[fid];
        if (data->dense_f[fid]) {
          if(start == 0 && end == data->n_data){
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[i];
              b_csw[bin_id].weight += is_weighted ? H[i] : 1;
            }
          }else{
            for(uint i = start;i < end;++i){
              auto bin_id = (fv[ids[i] >> 1] >> ((ids[i] & 1) << 2)) & 15;
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R_tmp[i];
              b_csw[bin_id].weight += is_weighted ? H_tmp[i] : 1;
            }
          }
        } else {
          std::vector<uint> &fi = (data->Xi)[fid];
          ushort j_unobserved = (data->data_header.unobserved_fv)[fid];

          // The following parallel for only works when fsz is very small
          // Remember to disable the outer parallel for before enable it, otherwise you will need to enable nested parallel for
          // #pragma omp parallel for schedule(static,1) reduction(vec_csw_plus:b_csw) if (fi.size() > 65536 * 8)
          for(int i = 0;i < fi.size();++i){
            if(in_leaf[fi[i]] == true){
              auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
              auto id = fi[i];
              b_csw[bin_id].count += 1;
              b_csw[bin_id].sum += R[id];
              b_csw[bin_id].weight += is_weighted ? H[id] : 1;

              b_csw[j_unobserved].sum -= R[id];
              b_csw[j_unobserved].count -= 1;
              b_csw[j_unobserved].weight -= is_weighted ? H[id] : 1;
            }
          }
          b_csw[j_unobserved].count += c_unobserved;
          b_csw[j_unobserved].sum += r_unobserved;
          b_csw[j_unobserved].weight += is_weighted ? h_unobserved : c_unobserved;
        }
      }
    }
    )

    setInLeaf<false>(start,end,ids);

  } else {
    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(guided),
      config->use_omp == true,
      for (int j = 0; j < fsz; ++j) {
        uint fid = (data->valid_fi)[(*fids)[j]];
        std::vector<HistBin> &b_csw = (*hist)[x][fid];
        int parent = nodes[x].parent;
        std::vector<HistBin> &pb_csw =
            (*hist)[parent][fid];
        std::vector<HistBin> &sb_csw = (*hist)[sib][fid];
        for (int k = 0; k < b_csw.size(); ++k) {
          b_csw[k].count = pb_csw[k].count - sb_csw[k].count;
          b_csw[k].sum = pb_csw[k].sum - sb_csw[k].sum;
          b_csw[k].weight =
              std::max((hist_t).0, pb_csw[k].weight - sb_csw[k].weight);
        }
      }
    )
  }
}

/**
 * Fit a decision tree to pseudo residuals which partitions the input space
 * into J disjoint regions and predicts a constant value for each region.
 * @param[in] ids: Pointer to sampled instance ids
 *            fids: Pointer to sampled feature ids
 * @post this->nodes and this->leaf_ids are populated.
 *       Feature importance is updated.
 */
void Tree::buildTree(std::vector<uint> *ids, std::vector<uint> *fids) {
  this->ids = (*(std::vector<uint> *)ids);
  this->fids = fids;
  in_leaf.resize(data->Y.size());
  fids_record = (*(std::vector<uint> *)fids);

  dense_fids.reserve(fids->size());
  sparse_fids.reserve(fids->size());
  for(int j = 0;j < fids->size();++j){
    int fid = (data->valid_fi)[(*fids)[j]];
    if(data->dense_f[fid])
      dense_fids.push_back(fid);
    else
      sparse_fids.push_back(fid);
  }


  nodes[0].idx = 0;
  nodes[0].start = 0;
  nodes[0].end = ids->size();

  int l, r;
  uint lsz, rsz, msz = config->tree_min_node_size, nrl = config->tree_n_random_layers;

  const int n_iter = n_leaves - 1;
  for (int i = 0; i < n_iter; ++i) {
    if (nrl == 0 && i == 0) trySplit(0, -1);
    if (i < Utils::ipow(2, nrl) - 1) {
      if (i == 0) binSort(i, -1);
      std::pair<int, int> fid_v = generateFidV(i);
      nodes[i].split_fi = fid_v.first;
      nodes[i].split_v = fid_v.second;
      nodes[i].is_random_node = true;

      if (fid_v.first == -1) continue;

      l = 2 * i + 1;
      r = l + 1;

      split(i, l);

      if (i >= Utils::ipow(2, nrl - 1) - 1) {
        if (lsz < rsz) {
          trySplit(l, -1);
          //trySplit(r, -1); is replaced by the subtraction
          trySplit(r, l);
        } else {
          trySplit(r, -1);
          //trySplit(l, -1);
          trySplit(l, r);
        }
      } else {
        if (lsz < rsz) {
          binSort(l, -1);
          binSort(r, l);
        } else {
          binSort(r, -1);
          binSort(l, r);
        }
      }

    } else {
      // find the node with max gain to split (calculated in trySplit)
      int idx = -1;
      double max_gain = -1;
      for (int j = 0; j < 2 * i + 1; ++j) {
        if (nodes[j].is_leaf && nodes[j].allow_build_subtree && nodes[j].gain > max_gain) {
          idx = j;
          max_gain = nodes[j].gain;
        }
      }

      l = 2 * i + 1;
      r = l + 1;

      if (idx == -1) {
//         fprintf(stderr, "[INFO] cannot split further.\n");
        break;
      }
      split(idx, l);
      lsz = nodes[l].end - nodes[l].start, rsz = nodes[r].end - nodes[r].start;

      if (lsz < msz && rsz < msz) {
//         fprintf(stderr,
//                 "[WARNING] Split is cancelled because of min node size!\n");
        continue;
      }

      // if(i + 1 < n_iter){
      if(i < n_iter){
        if (lsz < rsz) {
          trySplit(l, -1);
          //trySplit(r, -1); is replaced by the subtraction
          trySplit(r, l);
        } else {
          trySplit(r, -1);
          //trySplit(l, -1);
          trySplit(l, r);
        }
      }
    }
  }
  hist_record = *hist;
  regress();
  in_leaf.resize(0);
  in_leaf.shrink_to_fit();
}

void Tree::deleteIds() {
  std::vector<uint> unids_tmp = unids;

  std::map<int, std::vector<int*>> split_ptr;
  std::vector<uint> split_ids;

  int leaf_num = 0;
  for (auto node : nodes) {
    if (node.idx != -1 && node.is_leaf == true && node.start != node.end) leaf_num++;
  }

  int split_num = leaf_num + 1;
  split_ids.reserve(split_num);

  for (int i = 0; i < nodes.size(); i++) {
    if (nodes[i].start == -1 || nodes[i].end == -1) continue;
    split_ptr[nodes[i].start].push_back(&nodes[i].start);
    split_ptr[nodes[i].end].push_back(&nodes[i].end);
  }
  for (auto p : split_ptr) split_ids.emplace_back(p.first);

  std::vector<std::vector<uint>> id_global;
  std::vector<int> offset(split_num);
  std::vector<uint> id_records;
  id_global.resize(split_num);
  id_records.reserve(unids_tmp.size());

#pragma omp parallel for num_threads(config->n_threads)
  for (int i = 1; i < split_num; i++) {
    std::vector<uint>::iterator left_it, right_it, target_it;
    int left, right;
    left = split_ids[i - 1], right = split_ids[i];
    left_it = ids.begin() + left;
    right_it = ids.begin() + right;
    for (uint unid : unids_tmp) {
      target_it = std::lower_bound(left_it, right_it, unid); //
      if (target_it != right_it && *target_it == unid) {
        left_it = target_it;
        id_global[i].emplace_back(unid);
      }
    }
  }

  int sum_offset = 0, cnt = 0;
  for (int i = 0; i < split_num; i++) {
    id_records.insert(id_records.end(), id_global[i].begin(), id_global[i].end());
    sum_offset += id_global[i].size();
    offset[cnt] = sum_offset;
    cnt++;
  }

  for (int i = 0; i < split_num; i++) {
    for (int* e : split_ptr[split_ids[i]]) *e -= offset[i];
  }

  int last_ptr = 0, cur_ptr = 0, j = 0, ids_len = ids.size(), record_len = id_records.size();
  while (cur_ptr < ids_len) {
    if (j >= record_len || ids[cur_ptr] != id_records[j]) {
      ids[last_ptr] = ids[cur_ptr];
      last_ptr++;
      cur_ptr++;
    } else {
      cur_ptr++;
      j++;
    }
  }
  ids.resize(last_ptr);
}

void Tree::insertIds() {
  std::map<int, std::vector<int*>> split_ptr;
  std::vector<uint> split_ids, leaf_ids;

  for (auto node : nodes) {
    if (node.idx != -1 && node.is_leaf == true && node.start != node.end) {
      leaf_ids.emplace_back(node.idx);
    }
  }

  int leaf_num = leaf_ids.size();
  int split_num = leaf_num + 1;
  split_ids.reserve(split_num);

  for (int i = 0; i < nodes.size(); i++) {
    if (nodes[i].start == -1 || nodes[i].end == -1) continue;
    split_ptr[nodes[i].start].push_back(&nodes[i].start);
    split_ptr[nodes[i].end].push_back(&nodes[i].end);
  }
  // for (auto p : split_ptr) split_ids.emplace_back(p.first);

  std::vector<int> offsets(leaf_num);
  std::vector<uint> ids_new;
  std::vector<std::pair<int, int>> leaf_ids_sort(leaf_num); // first: split_num, second: leaf_ids's id

  for (int i = 0; i < leaf_num; i++) {
    int leaf_id = leaf_ids[i];
    offsets[i] = range[leaf_id].second - range[leaf_id].first;
    leaf_ids_sort[i].first = nodes[leaf_id].end;
    leaf_ids_sort[i].second = i;
  }
  std::sort(leaf_ids_sort.begin(), leaf_ids_sort.end());

  for (int i = 0; i < leaf_num; i++) {
    int leaf_ids_id = leaf_ids_sort[i].second;
    int leaf_id = leaf_ids[leaf_ids_id];
    ids_new.insert(ids_new.end(), \
            ids.begin() + nodes[leaf_id].start, \
            ids.begin() + nodes[leaf_id].end);
    ids_new.insert(ids_new.end(), \
            tune_ids.begin() + range[leaf_id].first, \
            tune_ids.begin() + range[leaf_id].second);
  }

  int total_offset = 0;
  for (int i = 0; i < leaf_num; i++) {
    int leaf_ids_id = leaf_ids_sort[i].second;
    int leaf_id = leaf_ids[leaf_ids_id];
    total_offset += offsets[leaf_ids_id];
    for (int* e : split_ptr[leaf_ids_sort[i].first]) *e += total_offset;
  }
  ids = ids_new;
}

void Tree::unlearnTree(std::vector<uint> *ids, std::vector<uint> *fids,
                       std::vector<uint> *unids_ptr) {

  this->unids = (*(std::vector<uint> *)unids_ptr);
  this->fids = fids;
  in_leaf.resize(data->Y.size());
  fids_record = (*(std::vector<uint> *)fids);

  deleteIds();

  dense_fids.reserve(fids->size());
  sparse_fids.reserve(fids->size());
  for(int j = 0;j < fids->size();++j){
    int fid = (data->valid_fi)[(*fids)[j]];
    if(data->dense_f[fid])
      dense_fids.push_back(fid);
    else
      sparse_fids.push_back(fid);
  }

  int n_nodes = nodes.size();
  std::vector<std::vector<uint>> retrain_subtrees;
  range = std::vector<std::pair<uint, uint>>(n_nodes, std::make_pair(0, 0));
  range[0] = std::make_pair(0, (this->unids).size());
#pragma omp parallel for
  for (uint i = 0; i < n_nodes; i++) nodes[i].allow_build_subtree = false;
  for (uint i = 0; i < n_nodes; i++) {
    auto& node = nodes[i];
    if (nodes[i].allow_build_subtree == true || nodes[i].idx < 0) continue;

    if (i == 0 || nodes[nodes[i].parent].is_random_node == true) {
      unlearnBinSort(i, -1, range[i].first, range[i].second, unids);
    } else if (i%2 == 1) {
      int lsz = nodes[i].end - nodes[i].start;
      int rsz = nodes[i + 1].end - nodes[i + 1].start;
      if (lsz <= rsz) {
        unlearnBinSort(i, -1, range[i].first, range[i].second, unids);
        if (range[i + 1].second - range[i + 1].first < config->data_max_n_bins) {
          unlearnBinSort(i + 1, -1, range[i + 1].first, range[i + 1].second, unids);
        } else {
          unlearnBinSort(i + 1, i, range[i + 1].first, range[i + 1].second, unids);
        }
      } else {
        unlearnBinSort(i + 1, -1, range[i + 1].first, range[i + 1].second, unids);
        if (range[i].second - range[i].first < config->data_max_n_bins) {
          unlearnBinSort(i, -1, range[i].first, range[i].second, unids);
        } else {
          unlearnBinSort(i, i + 1, range[i].first, range[i].second, unids);
        }
      }
    }
//     if (nodes[i].is_random_node == true) {
//       if (nodes[i].is_leaf == false) splitUnids(range, i, nodes[i].left);
//       continue;
//     }
    if (nodes[i].is_leaf == false) splitUnids(range, i, nodes[i].left);

    int best_fi = nodes[i].split_fi, best_v = nodes[i].split_v;
    double best_gain = -1;

    uint lsz, rsz, msz = config->tree_min_node_size;
    uint l = nodes[i].left, r = nodes[i].right;
    if (nodes[i].left != -1 && nodes[i].right != -1) {
      lsz = nodes[l].end - nodes[l].start, rsz = nodes[r].end - nodes[r].start;
    }
    if (nodes[i].is_leaf || (lsz != 0 && rsz != 0)) {
      if (nodes[i].is_random_node == true) continue;
      std::vector<SplitInfo> &splits = nodes[i].gains;
      std::vector<int> offsets(fids->size());
      int cnt = 0, splits_size = splits.size();
      for (int i = 1; i < splits_size; i++) {
        if (splits[i].split_fi != splits[i - 1].split_fi) offsets[cnt++] = i;
      }
      offsets[cnt] = splits_size;

      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1),
        config->use_omp == true,
        for (int j = 0; j < fids->size(); j++) {
          int gains_start = (j == 0?0:offsets[j - 1]), gains_end = offsets[j];
          int unids_start = range[i].first, unids_end = range[i].second;
          int fid = (data->valid_fi)[(*fids)[j]];

          featureGain(i, fid, splits, gains_start, gains_end, unids, unids_start, unids_end);
        }
      )

      for (int j = 0; j < splits.size(); j++) {
        auto &info = nodes[i].gains[j];
        if (info.gain > best_gain) {
          best_gain = info.gain;
          best_fi = info.split_fi;
          best_v = info.split_v;
        }
        nodes[i].gain = best_gain;
      }
      if (nodes[i].is_leaf == true) continue;
    }

    if (best_gain == -1 || !(best_fi == nodes[i].split_fi && best_v == nodes[i].split_v)) {
      std::vector<uint> retrain_ids;
      std::queue<uint> q;
      q.push(i);
      std::sort((this->ids).begin() + nodes[i].start, (this->ids).begin() + nodes[i].end);
      while (!q.empty()) {
        uint cur_id = q.front();
        q.pop();
        if (cur_id == -1) continue;
        nodes[cur_id].allow_build_subtree = true;
        retrain_ids.push_back(cur_id);
        int n_feats = data->data_header.n_feats, hist_size;
        for (uint j = 0; j < n_feats; ++j) {
          memset((*hist)[cur_id][j].data(), 0, sizeof(HistBin) * (*hist)[cur_id][j].size());
        }
        if (nodes[cur_id].is_leaf == false && nodes[cur_id].left != -1) q.push(nodes[cur_id].left);
        if (nodes[cur_id].is_leaf == false && nodes[cur_id].right != -1) q.push(nodes[cur_id].right);

        if (cur_id != i) nodes[cur_id] = TreeNode();
        else {
          auto& node = nodes[cur_id];
          node.is_leaf = true; // set retrained root's leaf as true
          // node.gains.clear();
          node.left = node.right = node.split_fi = node.split_v = node.gain = -1;
        }
      }
      std::sort(retrain_ids.begin(), retrain_ids.end());
      retrain_subtrees.emplace_back(retrain_ids);
    } else {
      // splitUnids(range, i, nodes[i].left);
    }
  }

  uint lsz, rsz, msz = config->tree_min_node_size;
  int l, r;
  for (std::vector<uint> &retrain_ids : retrain_subtrees) {
    int root = retrain_ids[0], n_iter = retrain_ids.size();
    trySplit(root, -1);
    for (int i = 1; i < n_iter; i += 2) {
      int idx = -1;
      double max_gain = -1;
      for (int j = 0; j < i; ++j) {
        if (nodes[retrain_ids[j]].is_leaf && nodes[retrain_ids[j]].gain > max_gain) {
          idx = retrain_ids[j];
          max_gain = nodes[retrain_ids[j]].gain;
        }
      }
      l = retrain_ids[i];
      r = retrain_ids[i + 1];
      if (idx == -1) {
//        fprintf(stderr, "[INFO] cannot split further.\n");
        break;
      }
      split(idx, l);
      nodes[l].has_retrain = true;
      nodes[r].has_retrain = true;
      lsz = nodes[l].end - nodes[l].start, rsz = nodes[r].end - nodes[r].start;
      if (lsz < msz && rsz < msz) {
//         fprintf(stderr,
//                 "[WARNING] Split is cancelled because of min node size!\n");
        continue;
      }

      // if(i + 1 < n_iter){
      if(i < n_iter){
        if (lsz < rsz) {
          trySplit(l, -1);
          //trySplit(r, -1); is replaced by the subtraction
          trySplit(r, l);
        } else {
          trySplit(r, -1);
          //trySplit(l, -1);
          trySplit(l, r);
        }
      }
    }
  }

  hist_record = *hist;
  regress(range);
  in_leaf.resize(0);
  in_leaf.shrink_to_fit();
}

void Tree::tuneTree(std::vector<uint> *ids, std::vector<uint> *fids,
                       std::vector<uint> *tune_ids_ptr) {
  this->tune_ids = (*(std::vector<uint> *)tune_ids_ptr);
  this->fids = fids;
  in_leaf.resize(data->Y.size());
  fids_record = (*(std::vector<uint> *)fids);

  dense_fids.reserve(fids->size());
  sparse_fids.reserve(fids->size());
  for(int j = 0;j < fids->size();++j){
    int fid = (data->valid_fi)[(*fids)[j]];
    if(data->dense_f[fid])
      dense_fids.push_back(fid);
    else
      sparse_fids.push_back(fid);
  }

  int n_nodes = nodes.size();
  std::vector<std::vector<uint>> retrain_subtrees;
  range = std::vector<std::pair<uint, uint>>(n_nodes, std::make_pair(0, 0));
  range[0] = std::make_pair(0, (this->tune_ids).size());

#pragma omp parallel for
  for (uint i = 0; i < n_nodes; i++) nodes[i].allow_build_subtree = false;
  for (uint i = 0; i < n_nodes; i++) {
    if (nodes[i].is_leaf == false && nodes[i].idx >= 0) splitIds(range, i, nodes[i].left, tune_ids);
  }
  insertIds();

  for (uint i = 0; i < n_nodes; i++) {
    auto& node = nodes[i];
    if (nodes[i].allow_build_subtree == true || nodes[i].idx < 0) continue;

    if (i == 0 || nodes[nodes[i].parent].is_random_node == true) {
      tuneBinSort(i, -1, range[i].first, range[i].second, tune_ids);
    } else if (i%2 == 1) {
      int lsz = nodes[i].end - nodes[i].start;
      int rsz = nodes[i + 1].end - nodes[i + 1].start;
      if (lsz <= rsz) {
        tuneBinSort(i, -1, range[i].first, range[i].second, tune_ids);
        if (range[i + 1].second - range[i + 1].first < config->data_max_n_bins) {
          tuneBinSort(i + 1, -1, range[i + 1].first, range[i + 1].second, tune_ids);
        } else {
          tuneBinSort(i + 1, i, range[i + 1].first, range[i + 1].second, tune_ids);
        }
      } else {
        tuneBinSort(i + 1, -1, range[i + 1].first, range[i + 1].second, tune_ids);
        if (range[i].second - range[i].first < config->data_max_n_bins) {
          tuneBinSort(i, -1, range[i].first, range[i].second, tune_ids);
        } else {
          tuneBinSort(i, i + 1, range[i].first, range[i].second, tune_ids);
        }
      }
    }
    if (nodes[i].is_random_node == true) continue;

    std::vector<SplitInfo> &splits = nodes[i].gains;
    std::vector<int> offsets(fids->size());
    int cnt = 0, splits_size = splits.size();
    for (int i = 1; i < splits_size; i++) {
      if (splits[i].split_fi != splits[i - 1].split_fi) offsets[cnt++] = i;
    }
    offsets[cnt] = splits_size;

    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(static, 1),
      config->use_omp == true,
      for (int j = 0; j < fids->size(); j++) {
        int gains_start = (j == 0?0:offsets[j - 1]), gains_end = offsets[j];
        int unids_start = range[i].first, unids_end = range[i].second;
        int fid = (data->valid_fi)[(*fids)[j]];

        featureGain(i, fid, splits, gains_start, gains_end, tune_ids, unids_start, unids_end);
      }
    )

    int best_fi = nodes[i].split_fi, best_v = nodes[i].split_v;
    double best_gain = -1;
    for (int j = 0; j < splits.size(); j++) {
      auto &info = nodes[i].gains[j];
      if (info.gain > best_gain) {
        best_gain = info.gain;
        best_fi = info.split_fi;
        best_v = info.split_v;
      }
      nodes[i].gain = best_gain;
    }
    if (nodes[i].is_leaf == true) continue;

    if (!(best_fi == nodes[i].split_fi && best_v == nodes[i].split_v)) {
      std::vector<uint> retrain_ids;
      std::queue<uint> q;
      q.push(i);
      std::sort((this->ids).begin() + nodes[i].start, (this->ids).begin() + nodes[i].end);
      while (!q.empty()) {
        uint cur_id = q.front();
        q.pop();
        if (cur_id == -1) continue;
        nodes[cur_id].allow_build_subtree = true;
        retrain_ids.push_back(cur_id);
        int n_feats = data->data_header.n_feats, hist_size;
        for (uint j = 0; j < n_feats; ++j) {
          memset((*hist)[cur_id][j].data(), 0, sizeof(HistBin) * (*hist)[cur_id][j].size());
        }
        if (nodes[cur_id].is_leaf == false && nodes[cur_id].left != -1) q.push(nodes[cur_id].left);
        if (nodes[cur_id].is_leaf == false && nodes[cur_id].right != -1) q.push(nodes[cur_id].right);

        if (cur_id != i) nodes[cur_id] = TreeNode();
        else {
          auto& node = nodes[cur_id];
          node.is_leaf = true; // set retrained root's leaf as true
          // node.gains.clear();
          node.left = node.right = node.split_fi = node.split_v = node.gain = -1;
        }
      }
      retrain_subtrees.emplace_back(retrain_ids);
    } else {
      ;
    }
  }

  uint lsz, rsz, msz = config->tree_min_node_size;
  int l, r;
  std::vector<bool> is_root(n_nodes);
  std::vector<uint> retrain_ids;
  for (int i = 1; i < n_nodes; i += 2) {
    if (nodes[i].idx == -1 && nodes[i + 1].idx == -1) {
      retrain_ids.emplace_back(i);
      retrain_ids.emplace_back(i + 1);
    }
  }

  if (retrain_ids.size() != 0) {
    for (int i = 0; i < retrain_subtrees.size(); i++) {
      int idx = retrain_subtrees[i][0];
      trySplit(idx, -1);
      retrain_ids.emplace_back(idx);
      is_root[idx] = true;
    }
    sort(retrain_ids.begin(), retrain_ids.end());
  }

  int n_iter = retrain_ids.size();
  for (int i = 0; i < n_iter; i += 2) {
    while (is_root[retrain_ids[i]] == true) {
      if (i < n_iter - 2) i++;
      else break;
    }
    int idx = -1;
    double max_gain = -1;
    for (int j = 0; j < i; ++j) {
      if (nodes[retrain_ids[j]].is_leaf && nodes[retrain_ids[j]].gain > max_gain) {
        idx = retrain_ids[j];
        max_gain = nodes[retrain_ids[j]].gain;
      }
    }
    if (max_gain < 0) continue;
    l = retrain_ids[i];
    r = retrain_ids[i + 1];
    if (idx == -1) {
//      fprintf(stderr, "[INFO] cannot split further.\n");
      break;
    }
    split(idx, l);
    nodes[l].has_retrain = true;
    nodes[r].has_retrain = true;
    lsz = nodes[l].end - nodes[l].start, rsz = nodes[r].end - nodes[r].start;
    if (lsz < msz && rsz < msz) {
//       fprintf(stderr,
//               "[WARNING] Split is cancelled because of min node size!\n");
      continue;
    }

    // if(i + 1 < n_iter){
    if(i < n_iter){
      if (lsz < rsz) {
        trySplit(l, -1);
        //trySplit(r, -1); is replaced by the subtraction
        trySplit(r, l);
      } else {
        trySplit(r, -1);
        //trySplit(l, -1);
        trySplit(l, r);
      }
    }
  }

  hist_record = *hist;
  regress(range);
  in_leaf.resize(0);
  in_leaf.shrink_to_fit();
}

void Tree::updateFeatureImportance(int iter) {
  for (double &x : (*feature_importance)) {
    x -= x / (iter + 1);
  }
  for (int i = 0; i < nodes.size(); ++i) {
    if (nodes[i].idx >= 0 && !nodes[i].is_leaf) {
      double tmp = nodes[i].gain / (iter + 1);
      if (tmp > 1e10) {
        tmp = 1e10;
      }
      (*feature_importance)[nodes[i].split_fi] += tmp;
    }
  }
}

/**
 * Compute the best split point for a feature at a node.
 * @param[in] x: Node id
 *            fid: Feature id
 */
std::pair<double, double> Tree::featureGain(int x, uint fid) const{
  auto &b_csw = (*hist)[x][fid];
  hist_t total_s = .0, total_w = .0;
  for (int i = 0; i < b_csw.size(); ++i) {
    total_s += b_csw[i].sum;
    total_w += b_csw[i].weight;
  }

  int l_c = 0, r_c = 0;
  hist_t l_w = 0, l_s = 0;
  int st = 0, ed = ((int)b_csw.size()) - 1;
  while (
      st <
      b_csw.size()) {  // st = min_i (\sum_{k <= i} counts[i]) >= min_node_size
    l_c += b_csw[st].count;
    l_s += b_csw[st].sum;
    l_w += b_csw[st].weight;
    if (l_c >= config->tree_min_node_size) break;
    ++st;
  }

  if (st == b_csw.size()) {
    return std::make_pair(-1, -1);
  }

  do {  // ed = max_i (\sum_{k > i} counts[i]) >= min_node_size
    r_c += b_csw[ed].count;
    ed--;
  } while (ed >= 0 && r_c < config->tree_min_node_size);

  if (st > ed) {
    return std::make_pair(-1, -1);
  }

  hist_t r_w = 0, r_s = 0;
  double max_gain = -1;
  int best_split_v = -1;
  for (int i = st; i <= ed; ++i) {
    if (b_csw[i].count == 0) {
      if (i + 1 < b_csw.size()) {
        l_w += b_csw[i + 1].weight;
        l_s += b_csw[i + 1].sum;
      }
      continue;
    }
    r_w = total_w - l_w;
    r_s = total_s - l_s;

    double gain = l_s / l_w * l_s + r_s / r_w * r_s;
    if (gain > max_gain /*&& gain < 1e10*/) {
      max_gain = gain;
      int offset = 1;
      while (i + offset < b_csw.size() && b_csw[i + offset].count == 0)
        offset++;
      best_split_v = i + offset / 2;
    }
    if (i + 1 < b_csw.size()) {
      l_w += b_csw[i + 1].weight;
      l_s += b_csw[i + 1].sum;
    }
  }

  max_gain -= total_s / total_w * total_s;
  return std::make_pair(max_gain, best_split_v);
}


void Tree::featureGain(int x, uint fid, std::vector<SplitInfo>& gains, int gains_start, int gains_end, \
                       std::vector<uint>& unids, int unids_start, int unids_end) { // Suppose unids is ordered
  if (unids_end - unids_start <= 0) return;
  const auto* H = hessian;
  const auto* R = residual;

  int unids_len = unids_end - unids_start;
  std::vector<hist_t> s(unids_len), w(unids_len);
  std::vector<int> bin_ids(unids_len, -1);

  if(data->auxDataWidth[fid] == 0){
    std::vector<data_quantized_t> &fv = (data->Xv)[fid];
    if (data->dense_f[fid]) {
      for(uint i = unids_start;i < unids_end;++i){
        int offset = i - unids_start;
        int id = unids[i];
        int bin_id = fv[id];
        
        bin_ids[offset] = bin_id;
        s[offset] = R[id];
        w[offset] = is_weighted ? H[id] : 1;
      }
    } else {
      std::vector<uint> &fi = (data->Xi)[fid];
      ushort j_unobserved = (data->data_header.unobserved_fv)[fid];
      for (int i = unids_start; i < unids_end; i++) {
        int offset = i - unids_start;
        auto id = unids[i];

        bin_ids[offset] = j_unobserved;
        s[offset] = R[id];
        w[offset] = is_weighted ? H[id] : 1;
      }
      for(int i = 0, j = unids_start; i < fi.size(); ++i){
        while (fi[i] > unids[j] && j < unids_end) j++;
        if (j >= unids_end) break;
        if (fi[i] == unids[j]) {
          int offset = j - unids_start;
          auto bin_id = fv[i];
          bin_ids[offset] = bin_id;
        }
      }
    }
  } else {
    std::vector<uint8_t> &fv = (data->auxData)[fid];
    if (data->dense_f[fid]) {
      for(uint i = unids_start;i < unids_end;++i){
        int offset = i - unids_start;
        int id = unids[i];
        int bin_id = (fv[id >> 1] >> ((id & 1) << 2)) & 15;
        
        bin_ids[offset] = bin_id;
        s[offset] = R[id];
        w[offset] = is_weighted ? H[id] : 1;
      }
    } else {
      std::vector<uint> &fi = (data->Xi)[fid];
      ushort j_unobserved = (data->data_header.unobserved_fv)[fid];
      for (int i = unids_start; i < unids_end; i++) {
        int offset = i - unids_start;
        auto id = unids[i];

        bin_ids[offset] = j_unobserved;
        s[offset] = R[id];
        w[offset] = is_weighted ? H[id] : 1;
      }
      for(int i = 0, j = unids_start; i < fi.size(); ++i){
        while (fi[i] > unids[j] && j < unids_end) j++;
        if (j >= unids_end) break;
        if (fi[i] == unids[j]) {
          int offset = j - unids_start;
          auto bin_id = (fv[i >> 1] >> ((i & 1) << 2)) & 15;
          bin_ids[offset] = bin_id;
        }
      }
    }
  }

  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(static, 1),
    config->use_omp == true,
    for (int i = gains_start; i < gains_end; i++) {
      if (gains[i].gain < 0) continue;
      int split_v = gains[i].split_v;
      double l_s = gains[i].l_s, l_w = gains[i].l_w, r_s = gains[i].r_s, r_w = gains[i].r_w;
      double delta_l_w = 0, delta_l_s = 0;
      for (int i = 0; i < unids_len; i++) {
        if (bin_ids[i] <= split_v) {
          delta_l_w += w[i];
          delta_l_s += s[i];
        }
      }
  
      double delta_r_w = 0, delta_r_s = 0;
      for (int i = 0; i < unids_len; i++) {
        if (bin_ids[i] > split_v) {
          delta_r_w += w[i];
          delta_r_s += s[i];
        }
      }

      double new_gain = -1;
      if (l_w != delta_l_w && r_w != delta_r_w) {
        double new_l_s = 0, new_r_s = 0, new_l_w = 0, new_r_w = 0;
        if (config->model_mode == "unlearn") {
          new_l_s = l_s - delta_l_s, new_r_s = r_s - delta_r_s;
          new_l_w = l_w - delta_l_w, new_r_w = r_w - delta_r_w;
        } else if (config->model_mode == "tune") {
          new_l_s = l_s + delta_l_s, new_r_s = r_s + delta_r_s;
          new_l_w = l_w + delta_l_w, new_r_w = r_w + delta_r_w;
        }

        new_gain = new_l_s/new_l_w * new_l_s + new_r_s/new_r_w * new_r_s;
        new_gain -= (new_l_s + new_r_s)/(new_l_w + new_r_w) * (new_l_s + new_r_s);

        gains[i].l_s = new_l_s;
        gains[i].l_w = new_l_w;
        gains[i].r_s = new_r_s;
        gains[i].r_w = new_r_w;
      }
      gains[i].gain = new_gain;
    }
  )
}

void Tree::featureGain(int x, uint fid, std::vector<SplitInfo>& gains, int start, int end) {
  auto &b_csw = (*hist)[x][fid];
  hist_t total_s = .0, total_w = .0;
  for (int i = 0; i < b_csw.size(); ++i) {
    total_s += b_csw[i].sum;
    total_w += b_csw[i].weight;
  }

  int l_c = 0, r_c = 0;
  hist_t l_w = 0, l_s = 0;
  int st = 0, ed = ((int)b_csw.size()) - 1;
  while (
      st <
      b_csw.size()) {  // st = min_i (\sum_{k <= i} counts[i]) >= min_node_size
    l_c += b_csw[st].count;
    l_s += b_csw[st].sum;
    l_w += b_csw[st].weight;
    if (l_c >= config->tree_min_node_size) break;
    ++st;
  }

  if (st == b_csw.size()) {
    return;
  }

  do {  // ed = max_i (\sum_{k > i} counts[i]) >= min_node_size
    r_c += b_csw[ed].count;
    ed--;
  } while (ed >= 0 && r_c < config->tree_min_node_size);

  if (st > ed) {
    return;
  }

  hist_t r_w = 0, r_s = 0;
  for (int i = st, j = start; i <= ed; ++i) {
    while(gains[j].split_v < st && j < end) j++;
    if (j >= end) break;
    if (i == gains[j].split_v) {
      r_w = total_w - l_w;
      r_s = total_s - l_s;

      double gain = l_s / l_w * l_s + r_s / r_w * r_s;
      gains[j].gain = gain;
      gains[j].l_s = l_s;
      gains[j].l_w = l_w;
      gains[j].r_s = r_s;
      gains[j].r_w = r_w;
      j++;
    }

    if (i + 1 < b_csw.size()) {
      l_w += b_csw[i + 1].weight;
      l_s += b_csw[i + 1].sum;
    }
  }

  double delta = total_s / total_w * total_s;
  for (int i = start; i < end; i++) {
    gains[i].gain -= delta;
  }
}

double Tree::featureGain(int x, uint fid, int split_v) const{
  auto &b_csw = (*hist)[x][fid];
  hist_t total_s = .0, total_w = .0;
  for (int i = 0; i < b_csw.size(); ++i) {
    total_s += b_csw[i].sum;
    total_w += b_csw[i].weight;
  }

  int l_c = 0, r_c = 0;
  hist_t l_w = 0, l_s = 0;
  int st = 0, ed = ((int)b_csw.size()) - 1;
  while (
      st <
      b_csw.size()) {  // st = min_i (\sum_{k <= i} counts[i]) >= min_node_size
    l_c += b_csw[st].count;
    l_s += b_csw[st].sum;
    l_w += b_csw[st].weight;
    ++st;
    if (l_c >= config->tree_min_node_size) break;
  }

  if (st == b_csw.size() || st > split_v) {
    return -1;
  }

  do {  // ed = max_i (\sum_{k > i} counts[i]) >= min_node_size
    r_c += b_csw[ed].count;
    ed--;
  } while (ed >= 0 && r_c < config->tree_min_node_size);

  if (st > ed || ed <= split_v) {
    return -1;
  }

  hist_t r_w = 0, r_s = 0;
  for (int i = st; i <= split_v; ++i) {
    if (i < b_csw.size()) {
      l_w += b_csw[i].weight;
      l_s += b_csw[i].sum;
    }
  }
  r_w = total_w - l_w;
  r_s = total_s - l_s;

  return l_s / l_w * l_s + r_s / r_w * r_s - total_s / total_w * total_s;
}

/**
 * Clear ids to save memory.
 */
void Tree::freeMemory() {
  ids.clear();
  ids.shrink_to_fit();
}

/**
 * Assign pointers before building a tree (for testing).
 */
void Tree::init(std::vector<std::vector<uint>> *l_buffer,
                std::vector<std::vector<uint>> *r_buffer) {
  this->l_buffer = l_buffer;
  this->r_buffer = r_buffer;
  n_threads = config->n_threads;
}

/**
 * Assign pointers before building a tree (for training).
 */
void Tree::init(
    std::vector<std::vector<std::vector<HistBin>>>
        *hist,
    std::vector<std::vector<uint>> *l_buffer,
    std::vector<std::vector<uint>> *r_buffer,
    std::vector<double> *feature_importance, double *hessian,
    double *residual,
                uint* ids_tmp,
                double* H_tmp,
                double* R_tmp) {
  this->hist = hist;
  if (hist == nullptr && (config->model_mode == "unlearn" || config->model_mode == "tune")) {
    this->hist = &hist_record;
  }
  this->l_buffer = l_buffer;
  this->r_buffer = r_buffer;
  this->feature_importance = feature_importance;
  this->hessian = hessian;
  this->residual = residual;
  n_threads = config->n_threads;
  this->ids_tmp = ids_tmp;
  this->H_tmp = H_tmp;
  this->R_tmp = R_tmp;
}

/**
 * Load nodes for a saved tree.
 * @param[in] fileptr: Pointer to the FILE object
 *            n_nodes: Number of nodes
 */
void Tree::populateTree(FILE *fileptr) {
  int ids_n = 0;
  size_t ret = fread(&ids_n, sizeof(ids_n), 1, fileptr);
  ids.resize(ids_n);
  for (int i = 0; i < ids_n; i++) {
    ret += fread(&ids[i], sizeof(ids[i]), 1, fileptr);
  }

  int n_nodes = 0;
  ret += fread(&n_nodes, sizeof(n_nodes), 1, fileptr);
  // use the actual tree size
  nodes.resize(n_nodes);

  int n_leafs = 0;
  int n = 0;
  for (n = 0; n < n_nodes; ++n) {
    TreeNode node = TreeNode();

    ret += fread(&node.idx, sizeof(node.idx), 1, fileptr);
    ret += fread(&node.parent, sizeof(node.parent), 1, fileptr);
    ret += fread(&node.left, sizeof(node.left), 1, fileptr);
    ret += fread(&node.right, sizeof(node.right), 1, fileptr);
    ret += fread(&node.start, sizeof(node.start), 1, fileptr);
    ret += fread(&node.end, sizeof(node.end), 1, fileptr);
    ret += fread(&node.split_fi, sizeof(node.split_fi), 1, fileptr);
    ret += fread(&node.split_v, sizeof(node.split_v), 1, fileptr);
    ret += fread(&node.gain, sizeof(node.gain), 1, fileptr);
    ret += fread(&node.predict_v, sizeof(node.predict_v), 1, fileptr);
    ret += fread(&node.is_random_node, sizeof(node.is_random_node), 1, fileptr);

    int gains_num = 0;
    ret += fread(&gains_num, sizeof(gains_num), 1, fileptr);
    node.gains.resize(gains_num);
    for (int i = 0; i < gains_num; i++) {
      SplitInfo info = SplitInfo();
      ret += fread(&info.split_fi, sizeof(info.split_fi), 1, fileptr);
      ret += fread(&info.gain, sizeof(info.gain), 1, fileptr);
      ret += fread(&info.split_v, sizeof(info.split_v), 1, fileptr);
      ret += fread(&info.l_s, sizeof(info.l_s), 1, fileptr);
      ret += fread(&info.l_w, sizeof(info.l_w), 1, fileptr);
      ret += fread(&info.r_s, sizeof(info.r_s), 1, fileptr);
      ret += fread(&info.r_w, sizeof(info.r_w), 1, fileptr);
      node.gains[i] = info;
    }

    // check whether a leaf
    if (node.idx < 0) {
      node.is_leaf = true;
    } else if (node.left == -1 && node.right == -1) {
      n_leafs++;
      leaf_ids.push_back(node.idx);
    } else {
      node.is_leaf = false;
    }

    nodes[n] = node;
  }

  int hist_size_1, hist_size_2, hist_size_3, fid;
  ret += fread(&hist_size_1, sizeof(hist_size_1), 1, fileptr);
  hist_record.resize(hist_size_1);
  for (int i = 0; i < hist_size_1; i++) {
    ret += fread(&hist_size_2, sizeof(hist_size_2), 1, fileptr);
    hist_record[i].resize(data->data_header.n_feats);
#pragma omp parallel for schedule(guided)
    for (int j = 0; j < data->data_header.n_feats; ++j) {
      hist_record[i][j].resize(data->data_header.n_bins_per_f[j]);
    }
    for (int j = 0; j < hist_size_2; j++) {
      ret += fread(&fid, sizeof(fid), 1, fileptr);
      hist_size_3 = data->data_header.n_bins_per_f[fid];
      hist_record[i][fid].resize(hist_size_3);
      for (int k = 0; k < hist_size_3; k++) {
        ret += fread(&hist_record[i][fid][k].count, \
               sizeof(hist_record[i][fid][k].count), 1, fileptr);
        ret += fread(&hist_record[i][fid][k].sum, \
               sizeof(hist_record[i][fid][k].sum), 1, fileptr);
        ret += fread(&hist_record[i][fid][k].weight, \
               sizeof(hist_record[i][fid][k].weight), 1, fileptr);
      }
    }
  }
}

/**
 * Predict region for a new instance.
 * @param[in] instance: All feature values of the instance
 * @return Region value
 */
double Tree::predict(std::vector<ushort> instance) {
  int i = 0;
  double predict_v;
  while (true) {
    // reach a leaf node
    if (nodes[i].is_leaf) {
      predict_v = nodes[i].predict_v;
      break;
    } else {  // go to left or right child
      i = instance[nodes[i].split_fi] <= nodes[i].split_v ? nodes[i].left
                                                          : nodes[i].right;
    }
  }
  return predict_v;
}

/**
 * Predict region for multiple instances.
 * @param[in] data: Dataset to train on
 * @return Region values for all instances.
 */
std::vector<double> Tree::predictAll(Data *data) {
  // use test data
  this->data = data;
  uint n_test = data->n_data;

  // initialize ids
  std::vector<uint> ids(n_test);
  std::iota(ids.begin(), ids.end(), 0);
  this->ids = ids;
  nodes[0].start = 0;
  nodes[0].end = n_test;

  std::vector<double> result(n_test, 0.0);

  std::queue<int> q;
  q.push(0);
  while (!q.empty()) {
    int i = q.front();
    q.pop();
    if (nodes[i].idx < 0) continue;
    if (!nodes[i].is_leaf) split(i, nodes[i].left);
    if (nodes[i].left != -1) q.push(nodes[i].left);
    if (nodes[i].right != -1) q.push(nodes[i].right);
  }

  // instances now distributed in each leaf
  // return corresponding region value for each
  for (auto lfid : leaf_ids) {
    int start = nodes[lfid].start, end = nodes[lfid].end;
    CONDITION_OMP_PARALLEL_FOR(
      omp parallel for schedule(static, 1),
      config->use_omp == true,
      for (int i = start; i < end; ++i)
        result[this->ids[i]] = nodes[lfid].predict_v;
    )
  }

  freeMemory();

  return result;
}

/**
 * Update region values for leaves.
 */
void Tree::regress() {
  double correction = 1.0;
  if (data->data_header.n_classes != 1 && config->model_name.size() >= 3 &&
      config->model_name.substr(0, 3) != "abc")
    correction -= 1.0 / data->data_header.n_classes;
  double upper = config->tree_clip_value, lower = -upper;

  auto* H = hessian;
  auto* R = residual;
  const bool is_weighted_update = config->model_use_weighted_update;
  leaf_ids.clear();

  for (int i = 0; i < nodes.size(); ++i) {
    if (nodes[i].idx >= 0 && nodes[i].is_leaf) {
      leaf_ids.push_back(i);
      double numerator = 0.0, denominator = 0.0;
      uint start = nodes[i].start, end = nodes[i].end;
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1) reduction(+: numerator, denominator),
        config->use_omp == true,
        for (uint d = start; d < end; ++d) {
          auto id = ids[d];
          numerator += R[id];
          denominator += H[id];
        }
      )
      nodes[i].predict_v =
          std::min(std::max(correction * numerator /
                                (denominator + config->tree_damping_factor),
                            lower),
                   upper);
    }
  }
}

void Tree::regress(std::vector<std::pair<uint, uint>>& range) {
  double correction = 1.0;
  if (data->data_header.n_classes != 1 && config->model_name.size() >= 3 &&
      config->model_name.substr(0, 3) != "abc")
    correction -= 1.0 / data->data_header.n_classes;
  double upper = config->tree_clip_value, lower = -upper;

  auto* H = hessian;
  auto* R = residual;
  const bool is_weighted_update = config->model_use_weighted_update;
  leaf_ids.clear();

  for (int i = 0; i < nodes.size(); ++i) {
    if (nodes[i].idx >= 0 && nodes[i].is_leaf) {
      leaf_ids.push_back(i);
      if (nodes[i].has_retrain == false && range[i].second - range[i].first <= 0) continue;
      double numerator = 0.0, denominator = 0.0;
      uint start = nodes[i].start, end = nodes[i].end;
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1) reduction(+: numerator, denominator),
        config->use_omp == true,
        for (uint d = start; d < end; ++d) {
          auto id = ids[d];
          numerator += R[id];
          denominator += H[id];
        }
      )
      nodes[i].predict_v =
          std::min(std::max(correction * numerator /
                                (denominator + config->tree_damping_factor),
                            lower),
                   upper);
    }
  }
}

/**
 * Save tree in a specified path.
 * @param[in] fp: Pointer to the FILE object
 */
void Tree::saveTree(FILE *fp) {
  int ids_n = ids.size();
  fwrite(&ids_n, sizeof(ids_n), 1, fp);
  for (uint &id : ids) {
    fwrite(&id, sizeof(id), 1, fp);
  }
  int n = nodes.size();
  fwrite(&n, sizeof(n), 1, fp);
  for (TreeNode &node : nodes) {
    fwrite(&node.idx, sizeof(node.idx), 1, fp);
    fwrite(&node.parent, sizeof(node.parent), 1, fp);
    fwrite(&node.left, sizeof(node.left), 1, fp);
    fwrite(&node.right, sizeof(node.right), 1, fp);
    fwrite(&node.start, sizeof(node.start), 1, fp);
    fwrite(&node.end, sizeof(node.end), 1, fp);
    fwrite(&node.split_fi, sizeof(node.split_fi), 1, fp);
    fwrite(&node.split_v, sizeof(node.split_v), 1, fp);
    fwrite(&node.gain, sizeof(node.gain), 1, fp);
    fwrite(&node.predict_v, sizeof(node.predict_v), 1, fp);
    fwrite(&node.is_random_node, sizeof(node.is_random_node), 1, fp);

    int gains_num = node.gains.size();
    fwrite(&gains_num, sizeof(gains_num), 1, fp);
    for (int i = 0; i < gains_num; i++) {
      fwrite(&node.gains[i].split_fi, sizeof(node.gains[i].split_fi), 1, fp);
      fwrite(&node.gains[i].gain, sizeof(node.gains[i].gain), 1, fp);
      fwrite(&node.gains[i].split_v, sizeof(node.gains[i].split_v), 1, fp);
      fwrite(&node.gains[i].l_s, sizeof(node.gains[i].l_s), 1, fp);
      fwrite(&node.gains[i].l_w, sizeof(node.gains[i].l_w), 1, fp);
      fwrite(&node.gains[i].r_s, sizeof(node.gains[i].r_s), 1, fp);
      fwrite(&node.gains[i].r_w, sizeof(node.gains[i].r_w), 1, fp);
    }
  }

  int hist_size_1 = hist_record.size();
  fwrite(&hist_size_1, sizeof(hist_size_1), 1, fp);
  for (int i = 0; i < hist_size_1; i++) {
    int hist_size_2 = fids_record.size();
    fwrite(&hist_size_2, sizeof(hist_size_2), 1, fp);
    for (int j = 0; j < hist_size_2; j++) {
      int fid = (data->valid_fi)[fids_record[j]];
      int hist_size_3 = hist_record[i][fid].size();
      fwrite(&fid, sizeof(fid), 1, fp);
      for (int k = 0; k < hist_size_3; k++) {
        fwrite(&hist_record[i][fid][k].count, \
               sizeof(hist_record[i][j][k].count), 1, fp);
        fwrite(&hist_record[i][fid][k].sum, \
               sizeof(hist_record[i][j][k].sum), 1, fp);
        fwrite(&hist_record[i][fid][k].weight, \
               sizeof(hist_record[i][j][k].weight), 1, fp);
      }
    }
  }
}

void Tree::splitUnids(std::vector<std::pair<uint, uint>>& range, int x, int l) {
  uint pstart = range[x].first;
  uint pend = range[x].second;
  uint n_ids = pend - pstart;
  if (n_ids <= 0) {
    range[l].first = pstart;
    range[l].second = pstart;
    range[l + 1].first = pstart;
    range[l + 1].second = pstart;
    return;
  }

  int split_v = nodes[x].split_v;
  uint fid = nodes[x].split_fi, li = pstart, ri = 0;
  std::vector<data_quantized_t> &fv = (data->Xv)[fid];
  std::vector<uint> unids_tmp_v(data->n_data);
  uint *unids_tmp = unids_tmp_v.data();

  if ((data->dense_f)[fid]) {
    if (config->use_omp && n_ids > n_threads && n_threads > 1) {
      uint buffer_sz = (n_ids + n_threads - 1) / n_threads;
      std::vector<int> left_is(n_threads, 0), right_is(n_threads, 0);
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1) reduction(+ : li),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          uint left_i = 0, right_i = 0;
          uint start = pstart + t * buffer_sz, end = start + buffer_sz;
          if (end > pend) end = pend;
          for (uint j = start; j < end; ++j) {
            uint unid = unids[j];
            if (fv[unid] <= split_v)
              (*l_buffer)[t][left_i++] = unid;
            else
              (*r_buffer)[t][right_i++] = unid;
          }

          left_is[t] = left_i;
          right_is[t] = right_i;
          li += left_i;
        }
      )

      std::vector<uint> left_sum(n_threads, 0), right_sum(n_threads, 0);
      for (int t = 1; t < n_threads; ++t) {
        left_sum[t] = left_sum[t - 1] + left_is[t - 1];
        right_sum[t] = right_sum[t - 1] + right_is[t - 1];
      }
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          std::move((*l_buffer)[t].begin(), (*l_buffer)[t].begin() + left_is[t],
                    unids.begin() + pstart + left_sum[t]);
          std::move((*r_buffer)[t].begin(), (*r_buffer)[t].begin() + right_is[t],
                    unids.begin() + li + right_sum[t]);
        }
      )
    } else {
      for (uint i = pstart; i < pend; ++i) {
        if(fv[unids[i]] <= split_v){
          unids[li] = unids[i];
          ++li;
        }else{
          unids_tmp[ri] = unids[i];
          ++ri;
        }
      }
      std::copy(unids_tmp, unids_tmp + ri, unids.begin() + li);
    }
  } else {
    int idx = 0;
    std::vector<uint> &fi = (data->Xi)[fid];
    int best_fsz = fi.size() - 1;
    ushort v, unobserved = (data->data_header.unobserved_fv)[fid];

    if (best_fsz >= 0) {
      for (uint i = pstart; i < pend; ++i) {
        uint id = unids[i];
        while (idx < best_fsz && fi[idx] < id) {
          ++idx;
        }
        v = (id == fi[idx]) ? fv[idx] : unobserved;
        if (v <= split_v) {
          unids[li++] = id;
        } else {
          unids_tmp[ri++] = id;
        }
      }
      std::copy(unids_tmp, unids_tmp + ri, unids.begin() + li);
    }
  }

  if (!(data->dense_f)[fid] && data->Xi[fid].size() <= 0) {
    li = ((data->data_header.unobserved_fv)[fid] <= split_v) ? pend : pstart;
  }
  range[l].first = pstart;
  range[l].second = li;
  range[l + 1].first = li;
  range[l + 1].second = pend;
}

void Tree::splitIds(std::vector<std::pair<uint, uint>>& range, int x, int l, std::vector<uint>& ids) {
  if (x < 0) return;
  uint pstart = range[x].first;
  uint pend = range[x].second;
  uint n_ids = pend - pstart;
  if (n_ids <= 0) {
    range[l].first = pstart;
    range[l].second = pstart;
    range[l + 1].first = pstart;
    range[l + 1].second = pstart;
    return;
  }

  int split_v = nodes[x].split_v;
  uint fid = nodes[x].split_fi, li = pstart, ri = 0;
  std::vector<data_quantized_t> &fv = (data->Xv)[fid];
  std::vector<uint> ids_tmp_v(data->n_data);
  uint *ids_tmp = ids_tmp_v.data();

  if ((data->dense_f)[fid]) {
    if (config->use_omp && n_ids > n_threads && n_threads > 1) {
      uint buffer_sz = (n_ids + n_threads - 1) / n_threads;
      std::vector<int> left_is(n_threads, 0), right_is(n_threads, 0);
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1) reduction(+ : li),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          uint left_i = 0, right_i = 0;
          uint start = pstart + t * buffer_sz, end = start + buffer_sz;
          if (end > pend) end = pend;
          for (uint j = start; j < end; ++j) {
            uint id = ids[j];
            if (fv[id] <= split_v)
              (*l_buffer)[t][left_i++] = id;
            else
              (*r_buffer)[t][right_i++] = id;
          }

          left_is[t] = left_i;
          right_is[t] = right_i;
          li += left_i;
        }
      )

      std::vector<uint> left_sum(n_threads, 0), right_sum(n_threads, 0);
      for (int t = 1; t < n_threads; ++t) {
        left_sum[t] = left_sum[t - 1] + left_is[t - 1];
        right_sum[t] = right_sum[t - 1] + right_is[t - 1];
      }
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          std::move((*l_buffer)[t].begin(), (*l_buffer)[t].begin() + left_is[t],
                    ids.begin() + pstart + left_sum[t]);
          std::move((*r_buffer)[t].begin(), (*r_buffer)[t].begin() + right_is[t],
                    ids.begin() + li + right_sum[t]);
        }
      )
    } else {
      for (uint i = pstart; i < pend; ++i) {
        if(fv[ids[i]] <= split_v){
          ids[li] = ids[i];
          ++li;
        }else{
          ids_tmp[ri] = ids[i];
          ++ri;
        }
      }
      std::copy(ids_tmp, ids_tmp + ri, ids.begin() + li);
    }
  } else {
    int idx = 0;
    std::vector<uint> &fi = (data->Xi)[fid];
    int best_fsz = fi.size() - 1;
    ushort v, unobserved = (data->data_header.unobserved_fv)[fid];

    if (best_fsz >= 0) {
      for (uint i = pstart; i < pend; ++i) {
        uint id = ids[i];
        while (idx < best_fsz && fi[idx] < id) {
          ++idx;
        }
        v = (id == fi[idx]) ? fv[idx] : unobserved;
        if (v <= split_v) {
          ids[li++] = id;
        } else {
          ids_tmp[ri++] = id;
        }
      }
      std::copy(ids_tmp, ids_tmp + ri, ids.begin() + li);
    }
  }

  if (!(data->dense_f)[fid] && data->Xi[fid].size() <= 0) {
    li = ((data->data_header.unobserved_fv)[fid] <= split_v) ? pend : pstart;
  }
  range[l].first = pstart;
  range[l].second = li;
  range[l + 1].first = li;
  range[l + 1].second = pend;
}



/**
 * Partition instances at a node into its left and right child.
 * @param[in] x: Node id
 *            l: Left child id
 * @post Order of ids is updated.
 *       Start/end are stored for left (node[l]) and right child (node[l+1]).
 */
void Tree::split(int x, int l) {
  uint pstart = nodes[x].start;
  uint pend = nodes[x].end;
  uint n_ids = pend - pstart;

  int split_v = nodes[x].split_v;
  uint fid = nodes[x].split_fi, li = pstart, ri = 0;
  std::vector<data_quantized_t> &fv = (data->Xv)[fid];


  nodes[x].is_leaf = false;
  nodes[x].allow_build_subtree = false;
  nodes[x].left = l;
  nodes[x].right = l + 1;
  nodes[l].idx = l;
  nodes[l].parent = x;
  nodes[l + 1].idx = l + 1;
  nodes[l + 1].parent = x;

  if ((data->dense_f)[fid]) {
    if (config->use_omp && n_ids > n_threads && n_threads > 1) {
      uint buffer_sz = (n_ids + n_threads - 1) / n_threads;
      std::vector<int> left_is(n_threads, 0), right_is(n_threads, 0);
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1) reduction(+ : li),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          uint left_i = 0, right_i = 0;
          uint start = pstart + t * buffer_sz, end = start + buffer_sz;
          if (end > pend) end = pend;
          for (uint j = start; j < end; ++j) {
            uint id = ids[j];
            if (fv[id] <= split_v)
              (*l_buffer)[t][left_i++] = id;
            else
              (*r_buffer)[t][right_i++] = id;
          }

          left_is[t] = left_i;
          right_is[t] = right_i;
          li += left_i;
        }
      )

      std::vector<uint> left_sum(n_threads, 0), right_sum(n_threads, 0);
      for (int t = 1; t < n_threads; ++t) {
        left_sum[t] = left_sum[t - 1] + left_is[t - 1];
        right_sum[t] = right_sum[t - 1] + right_is[t - 1];
      }
      CONDITION_OMP_PARALLEL_FOR(
        omp parallel for schedule(static, 1),
        config->use_omp == true,
        for (int t = 0; t < n_threads; ++t) {
          std::move((*l_buffer)[t].begin(), (*l_buffer)[t].begin() + left_is[t],
                    ids.begin() + pstart + left_sum[t]);
          std::move((*r_buffer)[t].begin(), (*r_buffer)[t].begin() + right_is[t],
                    ids.begin() + li + right_sum[t]);
        }
      )
    } else {
      for (uint i = pstart; i < pend; ++i) {
        if(fv[ids[i]] <= split_v){
          ids[li] = ids[i];
          ++li;
        }else{
          ids_tmp[ri] = ids[i];
          ++ri;
        }
      }
      std::copy(ids_tmp, ids_tmp + ri, ids.begin() + li);
    }
  } else {
    int idx = 0;
    std::vector<uint> &fi = (data->Xi)[fid];
    int best_fsz = fi.size() - 1;
    ushort v, unobserved = (data->data_header.unobserved_fv)[fid];

    if (best_fsz >= 0) {
      for (uint i = pstart; i < pend; ++i) {
        uint id = ids[i];
        while (idx < best_fsz && fi[idx] < id) {
          ++idx;
        }
        v = (id == fi[idx]) ? fv[idx] : unobserved;
        if (v <= split_v)
          ids[li++] = id;
        else
          ids_tmp[ri++] = id;
      }
      std::copy(ids_tmp, ids_tmp + ri, ids.begin() + li);
    }
  }

  if (!(data->dense_f)[fid] && data->Xi[fid].size() <= 0) {
    li = ((data->data_header.unobserved_fv)[fid] <= split_v) ? pend : pstart;
  }
  nodes[l].start = pstart;
  nodes[l].end = li;
  nodes[l + 1].start = li;
  nodes[l + 1].end = pend;
}

/**
 * Compute the best feature to split as well as its information gain.
 * Meanwhile, store the bin sort results for later.
 * @param[in] x: Node id
 *            sib: Sibling id
 * @post gain, split_fi, and split_v are stored for node[x].
 */
void Tree::trySplit(int x, int sib) {
  binSort(x, sib);

  if ((nodes[x].end - nodes[x].start) < config->tree_min_node_size) return;

  std::vector<SplitInfo> gains;
  std::vector<int> offsets(fids->size());
  nodes[x].gains.clear();

  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(guided),
    config->use_omp == true,
    for (int j = 0; j < fids->size(); ++j) {
        int fid = (data->valid_fi)[(*fids)[j]];
        int n_bins = data->data_header.n_bins_per_f[fid];
        int offset = n_bins * config->feature_split_sample_rate;
	if (config->feature_split_sample_rate >= 1 - 1e-5) offset = n_bins;
        else if (offset < 1) offset = std::min(1, n_bins);
        offsets[j] = offset;
    }
  )
  for (int j = 1; j < fids->size(); ++j) offsets[j] += offsets[j - 1];
  gains.resize(*offsets.rbegin());

  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(guided),
    config->use_omp == true,
    for (int j = 0; j < fids->size(); ++j) {
      int fid = (data->valid_fi)[(*fids)[j]];
      int n_bins = data->data_header.n_bins_per_f[fid];
      int start = (j == 0?0:offsets[j - 1]), end = offsets[j];
      if (end - start == n_bins) {
        for (int i = start; i < end; i++) {
          gains[i].split_fi = fid;
          gains[i].split_v = i - start;
        }
      }
    }
  )

  for (int j = 0; j < fids->size(); ++j) {
    int fid = (data->valid_fi)[(*fids)[j]];
    int n_bins = data->data_header.n_bins_per_f[fid];
    int start = (j == 0?0:offsets[j - 1]), end = offsets[j];
    if (end - start != n_bins) {
      std::vector<uint> sample_results = sample(n_bins, end - start);
      for (int i = start; i < end; i++) {
        gains[i].split_fi = fid;
        gains[i].split_v = sample_results[i - start];
      }
    }
  }

  CONDITION_OMP_PARALLEL_FOR(
    omp parallel for schedule(guided),
    config->use_omp == true,
    for (int j = 0; j < fids->size(); ++j) {
        int fid = (data->valid_fi)[(*fids)[j]];
        int start = (j == 0?0:offsets[j - 1]), end = offsets[j];
        featureGain(x, fid, gains, start, end);
    }
  )

  SplitInfo best_info;
  best_info.gain = -1;

  for(int j = 0;j < gains.size();++j){
    const auto& info = gains[j];
    if (info.gain > best_info.gain) {
      best_info = info;
    }
  }

  if (best_info.gain < 0) return;
  nodes[x].gain = best_info.gain;
  nodes[x].split_fi = best_info.split_fi;
  nodes[x].split_v = best_info.split_v;
  nodes[x].gains = gains;
}

std::vector<unsigned int> Tree::sample(int n, int n_samples) {

  std::vector<unsigned int> indices(n_samples);
  std::iota(indices.begin(), indices.end(), 0);  // set values
  std::vector<bool> bool_indices(n, false);
  std::fill(bool_indices.begin(), bool_indices.begin() + n_samples, true);
  for (int i = n_samples + 1; i <= n; ++i) {
    int gen = (rand() % (i)) + 1;  // determine if need to swap
    if (gen <= n_samples) {
      int swap = rand() % n_samples;
      bool_indices[indices[swap]] = false;
      indices[swap] = i - 1;
      bool_indices[i - 1] = true;
    }
  }
  int index = 0;
  for (int i = 0; i < n; i++) {  // populate indices with sorted samples
    if (bool_indices[i]) {
      indices[index] = i;
      ++index;
    }
  }
  return indices;
}

std::pair<int, int> Tree::getValidRange(int x, uint fid) {
  auto &b_csw = (*hist)[x][fid];

  int l_c = 0, r_c = 0;
  int st = 0, ed = ((int)b_csw.size()) - 1;
  while (
      st <
      b_csw.size()) {  // st = min_i (\sum_{k <= i} counts[i]) >= min_node_size
    l_c += b_csw[st].count;
    if (l_c >= config->tree_min_node_size) break;
    ++st;
  }

  if (st == b_csw.size()) {
    return std::make_pair(-1, -1);
  }

  do {  // ed = max_i (\sum_{k > i} counts[i]) >= min_node_size
    r_c += b_csw[ed].count;
    ed--;
  } while (ed >= 0 && r_c < config->tree_min_node_size);

  if (st > ed) {
    return std::make_pair(-1, -1);
  }

  return std::make_pair(st, ed);
}

std::pair<int, int> Tree::generateFidV(int x) {
  std::vector<std::pair<int, std::pair<int, int>>> valid_ranges;

  std::pair<int, int> range;
  for (int j = 0; j < fids->size(); ++j) {
    int fid = (data->valid_fi)[(*fids)[j]];
    range = getValidRange(x, fid);
    if (range.first == -1) continue;
    valid_ranges.emplace_back(std::make_pair(fid, range));
  }

  if (valid_ranges.size() == 0) {
    return std::make_pair(-1, -1);
  }

  int random_id = sample(valid_ranges.size(), 1)[0];
  int random_fid = valid_ranges[random_id].first;
  int st = valid_ranges[random_id].second.first;
  int ed = valid_ranges[random_id].second.second;
  int random_v = sample(ed - st + 1, 1)[0] + st;
  return std::make_pair(random_fid, random_v);
}

}  // namespace ABCBoost
