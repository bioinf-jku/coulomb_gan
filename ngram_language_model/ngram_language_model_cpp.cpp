#include "ngram_language_model_cpp.h"
#include <assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

namespace language {

using namespace std;

NgramLanguageModelCPP::NgramLanguageModelCPP(int n_in, int m_in)
    : n(n_in),
      m(m_in),
      tree_memory_sparse(0),
      tree_memory_dense(0),
      total_counts(n_in, 0),
      root_node(new DenseNode(m_in)),
      unique_ngrams(n_in, vector<vector<int> >(0)),
      precalc_log_like(n_in, vector<double>(0)) {
}

void NgramLanguageModelCPP::add_sample(const vector<int>& sample) {
  Node* current_node;
  for (int i = 0; i < n; ++i) {
    total_counts[i] += max(static_cast<int>(sample.size()) - i, 0);
  }
  for (int i = 0; i < static_cast<int>(sample.size()); ++i) {
    current_node = root_node.get();
    for (int j = i; j < min(i+n, static_cast<int>(sample.size())); ++j) {
      if(!current_node->get_child(sample[j])) {
        Node* next_node;
        if (j - i > 3) {
          next_node = new SparseNode(m);
          tree_memory_sparse += sizeof(shared_ptr<Node>) + sizeof(SparseNode);
        } else {
          next_node = new DenseNode(m);
          tree_memory_dense += m * sizeof(nullptr) + sizeof(shared_ptr<Node>) + sizeof(DenseNode);
        }
        current_node->add_child(sample[j], next_node);
        
        vector<int> v(sample.cbegin() + i, sample.cbegin() + j + 1);
        unique_ngrams[j-i].push_back(v);
      }
      current_node = current_node->get_child(sample[j]);
      current_node->increment_count();
    }
  }
  // now the precalculated log likelihoods are off, reset them
  precalc_log_like = vector<vector<double> >(n, vector<double>(0));
}

const vector<vector<int> > NgramLanguageModelCPP::get_unique_ngrams(int ngram_length) const {
  return unique_ngrams[ngram_length - 1];
}

const double NgramLanguageModelCPP::js_with(const NgramLanguageModelCPP& other, int ngram_length) const {
  assert(n == other.n);
  // precalculate log likelihoods if need be
  if (!precalc_log_like[ngram_length-1].size())
    precalc_log_like[ngram_length-1] = log_likelihood(get_unique_ngrams(ngram_length));
  if (!other.precalc_log_like[ngram_length-1].size())
    other.precalc_log_like[ngram_length-1] = other.log_likelihood(other.get_unique_ngrams(ngram_length));

  vector<double> cross_likelihood = log_likelihood(other.get_unique_ngrams(ngram_length));
  double kl_p_m = inner_product(other.precalc_log_like[ngram_length-1].cbegin(),
                                other.precalc_log_like[ngram_length-1].cend(),
                                cross_likelihood.cbegin(),
                                0.,
                                std::plus<double>(),
                                [](double x, double y){return exp(x) * (x - (log(exp(x-log(2.)) + exp(y-log(2.))))); });

  cross_likelihood = other.log_likelihood(get_unique_ngrams(ngram_length)); 
  double kl_q_m = inner_product(precalc_log_like[ngram_length-1].cbegin(),
                                precalc_log_like[ngram_length-1].cend(),
                                cross_likelihood.cbegin(),
                                0.,
                                std::plus<double>(),
                                [](double x, double y){return exp(x) * (x - (log(exp(x-log(2.)) + exp(y-log(2.))))); });

  return 0.5*(kl_p_m + kl_q_m) / log(2.);
}

const double NgramLanguageModelCPP::log_likelihood(const vector<int>& ngram) const {
  Node* current_node = root_node.get();
  const int ngram_length = static_cast<int>(ngram.size());
  for(int i = 0; i < ngram_length; ++i) {
    current_node = current_node->get_child(ngram[i]);
    if(!current_node)
      return -1 * numeric_limits<double>::infinity();
  }
  return log(current_node->get_count()) - log(total_counts[ngram_length-1]);
}

const vector<double> NgramLanguageModelCPP::log_likelihood(const vector<vector<int> >& ngrams) const {
  vector<double> answer(ngrams.size());
  for(int i = 0; i < static_cast<int>(ngrams.size()); ++i) {
    answer[i] = log_likelihood(ngrams[i]);
  } 
  return answer;
}

const long int NgramLanguageModelCPP::get_memory() const {
  // precalculate log likelihoods if need be
  for(int i = 0; i < n; ++i) {
    if (!precalc_log_like[i].size())
      precalc_log_like[i] = log_likelihood(get_unique_ngrams(i+1));
  }
  long int total_counts_memory = total_counts.size() * sizeof(total_counts[0]) + sizeof total_counts;
  std::cout << "smart pointer size = " << sizeof(shared_ptr<Node>) << std::endl;
  std::cout << "total count.size " << total_counts.size() << std::endl;
  std::cout << "total counts = " << sizeof total_counts << std::endl;
  std::cout << "total_counts_memory " << total_counts_memory << std::endl;
  
  long int precalc_log_like_memory = sizeof precalc_log_like;
  for(int i = 0; i < precalc_log_like.size(); ++i) {
    precalc_log_like_memory += sizeof precalc_log_like[i];
    if (precalc_log_like[i].size() > 0)
      precalc_log_like_memory += precalc_log_like[i].size() * sizeof(precalc_log_like[i][0]);
  }
  std::cout << "precalc_log_like_memory " << precalc_log_like_memory << std::endl;
  
  long int unique_ngrams_memory = sizeof unique_ngrams;
  for(int i = 0; i < unique_ngrams.size(); ++i) {
    unique_ngrams_memory += sizeof unique_ngrams[i];
    for(int j = 0; j < unique_ngrams[i].size(); ++j) {
      unique_ngrams_memory += sizeof unique_ngrams[i][j];
      if (unique_ngrams[i][j].size() > 0)
        unique_ngrams_memory += unique_ngrams[i][j].size() * sizeof(unique_ngrams[i][j][0]);
    }
  }
  std::cout << "unique_ngrams_memory " << unique_ngrams_memory << std::endl;

  std::cout << "tree_memory_dense = " << tree_memory_dense << std::endl;
  std::cout << "tree_memory_sparse = " << tree_memory_sparse << std::endl;
  
  
  
  return 0;
}


}  // namespace language
