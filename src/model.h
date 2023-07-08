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

#ifndef ABCBOOST_MODEL_H
#define ABCBOOST_MODEL_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "config.h"
#include "data.h"
#include "tree.h"

namespace ABCBoost {

struct ModelHeader {
  // config info
  Config config;
	DataHeader auxDataHeader; // saves mapping

  void serialize(FILE* fp) { 
		config.serialize(fp);
		if(config.no_map == true)
			auxDataHeader.serialize_no_map(fp);
    else
			auxDataHeader.serialize(fp);
	}

  static ModelHeader deserialize(FILE* fp) {
    ModelHeader model_header;
    model_header.config = Config::deserialize(fp);
		if(model_header.config.no_map == true)
			model_header.auxDataHeader = DataHeader::deserialize_no_map(fp);
    else
			model_header.auxDataHeader = DataHeader::deserialize(fp);
    return model_header;
  }
};

class GradientBoosting {
	
 protected:
  // [n_nodes, n_feats, :] of (count, sum, weight)
  std::vector<std::vector<std::vector<HistBin>>> hist;
  std::vector<std::vector<std::unique_ptr<Tree>>> additive_trees;
  std::vector<std::vector<double>> F;//, hessians, residuals;
	std::vector<double> hessians,residuals;
  std::vector<std::vector<std::vector<double>>> F_record;
  std::vector<std::vector<double>> hessians_record, residuals_record;
  std::vector<double> feature_importance;
  std::vector<unsigned int> ids, fids, unids;
  std::vector<std::vector<unsigned int>> fids_record;
  std::string experiment_path;

  Config* config;
  Data* data;
  FILE* log_out;
  bool sample_data, sample_feature;
	
	std::vector<double> R_tmp;
	std::vector<double> H_tmp;
	std::vector<uint> ids_tmp;

  virtual void saveF();

 public:
  GradientBoosting(Data* data, Config* config);
  virtual ~GradientBoosting();

  int start_epoch = 0;
  std::vector<std::vector<double>> testlog;

  int argmax(std::vector<double>& f_vector);
  virtual double getAccuracy();
	virtual int getError();
  int getError(std::vector<std::vector<double>>& F);
  virtual double getLoss();
  virtual double getAUC();
  double getAUC(double* f_values, int k);
  void getTopFeatures();
  virtual void init();
  std::vector<unsigned int> sample(int n, double sample_rate);
  void setupExperiment();
  void softmax(std::vector<double>& v);
  void updateF(int k, Tree* currTree);
  void updateF(int m, int k, Tree *tree, \
      std::vector<std::vector<std::vector<double>>>& F);
  void computeHessianResidual(std::vector<std::vector<double>>& F);
  void zeroBins();

  virtual int loadModel();
  void deleteIds(std::vector<uint>& unids, std::vector<uint>& ids2ids);
  void alignIdsAfterUnlearning(std::vector<uint>& ids2ids);
  virtual void saveModel(int iter);
  virtual void saveData(std::vector<uint>* unids_ptr = nullptr);
  virtual void test();
  virtual void train();
  virtual void unlearn(std::vector<uint>& unidxs);
  virtual void tune(std::vector<uint>& tune_ids);

  virtual void savePrediction();

  virtual void returnPrediction(double* prediction,double* probability);

  Config* getConfig() { return config; }
  Data* getData() { return data; }
  void setExperimentPath(std::string path) { experiment_path = path; }
  void printF() {
    for (int i = 0; i < 5; ++i) printf("F[0][%d]: %f\n", i, F[0][i]);
  }
  static ModelHeader loadModelHeader(Config* config);
  std::vector<std::vector<std::vector<unsigned int>>> initBuffer();

  void serializeTrees(FILE* fp, int M);
  void deserializeTrees(FILE* fp);
	void print_test_message(int iter,double iter_time,int& low_err);
	virtual void print_test_message(int iter,double iter_time,double& low_loss) {}
	virtual void print_train_message(int iter,double loss,double iter_time);
        void print_unlearn_message(int iter,double loss,double iter_time, std::vector<std::vector<double>>& F);
	
  // only for ranking
  virtual void print_rank_test_message(int iter,double iter_time);
  virtual void print_rank_train_message(int iter,double NDCG,double iter_time);
  std::pair<double,double> getNDCG();
};

class Regression : public GradientBoosting {
 public:
  Regression(Data* data, Config* config);
  void test();
  void train();
  void init();

 private:
	virtual void print_test_message(int iter,double iter_time,double& low_loss);
	virtual void print_train_message(int iter,double loss,double iter_time);
  void computeHessianResidual();
  double getLSLoss();
  double getL1Loss();
  double getLpLoss(const double p);
  double getHuberLoss();
  int loadModel();
  void saveModel(int iter);
};

class BinaryMart : public GradientBoosting {
 public:
  BinaryMart(Data* data, Config* config);
  void test();
  void train();
  void init();
	void updateF(Tree* tree);
	double getLoss();
  double getAUC();
  void savePrediction();
  void returnPrediction(double* prediction,double* probability);
  
	std::vector<double> F;//, hessians, residuals;

 private:
  void computeHessianResidual();
	double getAccuracy();
	int getError();
};

class Mart : public GradientBoosting {
 public:
  Mart(Data* data, Config* config);
  void test();
  void train();
  void unlearn(std::vector<uint>& unidxs);
  void tune(std::vector<uint>& tune_ids);
  void test_rank();
  friend class MOCMart;

 private:
  void computeHessianResidual();
  void computeHessianResidual(std::vector<std::vector<double>>& F);
};

class ABCMart : public GradientBoosting {
 public:
  ABCMart(Data* data, Config* config);
  void train_worst();
  void test();
  void train();
  void init();

 private:
  std::vector<int> base_classes;
  std::vector<double> class_losses;
  void computeHessianResidual();
  void computeHessianResidual(int b);
	void computeHessianResidual(int b,std::vector<unsigned int>& abc_sample_ids);
  double getLoss();
  void updateBaseFValues(int b);
  int loadModel();
  int loadModelWithoutData();
  void saveModel(int iter);
	double getLossRaw(int b,const std::vector<unsigned int>& ids) const;
	void updateNormF(int k, Tree *tree);
};

class LambdaMart : public GradientBoosting {
 public:
  LambdaMart(Data* data, Config* config);
  void test();
  void train();
  void savePrediction();

 private:
  void computeHessianResidual();
};

class GBRank : public GradientBoosting {
 public:
  GBRank(Data* data, Config* config);
  void test();
  void train();
  void savePrediction();

 private:
  double tau = 0.1;
  double tau2 = tau;
  void computeHessianResidual();
  void GBupdateF(int k, Tree* currTree,int n_iter);
};

}  // namespace ABCBoost

#endif  // ABCBOOST_MODEL_H

