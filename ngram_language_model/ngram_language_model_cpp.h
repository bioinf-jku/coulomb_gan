#include <map>
#include <memory>
#include <vector>

namespace language {

using namespace std;

class Node {
public:
  Node() : count(0) {}
  Node(int alphabet_size) : count(0) {}
  virtual ~Node() {}
  virtual void add_child(int lookup, Node* node) = 0;
  /// find a child, return a nullptr if it doesn't exits (yet)
  virtual Node* get_child(int lookup) = 0;
  void set_count(int in_count) {count = in_count;}
  int get_count() const {return count;}
  void increment_count() {++count;}
private:
  /// the number of times this ngram has made an appearance
  int count;
};

class DenseNode : public Node {
public:
  /// construct a node with a bunch of nullptr children
  DenseNode(int alphabet_size)
    : Node(alphabet_size), children(alphabet_size, nullptr) {
    children.shrink_to_fit();
  }
  /// easy destructor because smart pointers FTW
  virtual ~DenseNode() {}
  virtual void add_child(int lookup, Node* node) {
    children[lookup] = shared_ptr<Node>(node);
  }
  virtual Node* get_child(int lookup) {
    return children[lookup].get();
  }
private:
  /// forbid default constructor
  DenseNode() {}
  /// if you love pain, then please, don't use smart pointers
  vector<shared_ptr<Node> > children;
};

class SparseNode : public Node {
public:
  /// construct a node with a bunch of nullptr children
  SparseNode(int alphabet_size)
    : Node(alphabet_size), children() {
  }
  /// easy destructor because smart pointers FTW
  virtual ~SparseNode() {}
  virtual void add_child(int lookup, Node* node) {
    children[lookup] = shared_ptr<Node>(node);
  }
  virtual Node* get_child(int lookup) {
    auto search = children.find(lookup);
    if(search != children.end()) {
      return search->second.get();
    } else {
      return nullptr;
    }
  }
private:
  /// forbid default constructor
  SparseNode() {}
  /// if you love pain, then please, don't use smart pointers
  map<int, shared_ptr<Node> > children;
};

class NgramLanguageModelCPP {
public:
  /**
   * @param n the number of ngrams to construct
   * @param m size of alphabet
   */
  NgramLanguageModelCPP(int n_in, int m_in);
  virtual ~NgramLanguageModelCPP() {};
  /**
   * Add another line of text to the language model
   * @param samples vector of ints containing the sample
   */
  void add_sample(const vector<int>& samples);

  /**
   * Return all unique ngrams of length n
   */
  const vector<vector<int> > get_unique_ngrams(int ngram_length) const;

  /**
   * Calculate the js distance between this model and another one
   */
  const double js_with(const NgramLanguageModelCPP& other, int ngram_length) const;

  /**
   * calculate the log likelihood for either a single ngram or a list of ngrams
   */
  const double log_likelihood(const vector<int>& ngram) const;
  const vector<double> log_likelihood(const vector<vector<int> >& ngrams) const;
  /// Get the memory that this object consumes
  const long int get_memory() const;
private:
  ///Forbid defalut construction
  NgramLanguageModelCPP() : n(0), m(0) {};

  /// which ngram sizes should be saved
  const int n;

  /// size of possible alphabet
  const int m;

  /// total number of ngrams for each possible ngram length
  vector<long int> total_counts;

  /// all ngrams are in a tree structure, this is the root node
  shared_ptr<DenseNode> root_node;

  /// list of all unique ngrams of different lengths
  vector<vector<vector<int> > > unique_ngrams;

  /// precalculate the log likelihood of the ngrams in this model
  mutable vector<vector<double> > precalc_log_like;
  
  long int tree_memory_dense;
  long int tree_memory_sparse;
};

}  // namespace language
