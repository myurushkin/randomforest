#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <valarray>
#include <boost/serialization/valarray.hpp>
#include <memory>

class Dataset
{
    struct SharedData
    {
        std::valarray<double> data;
        std::valarray<int> sortInd;

        std::valarray<int> indBuffer;
        std::vector<std::pair<int, double> > orderBuffer;
    };

public:
    Dataset();
    Dataset(int rowNum, int columnNum);

    void set_data(int rowNum, int columnNum, const std::valarray<double>& data);
    void set_newSize(int rowNum, int columnNum);
    void prepare();

    double &operator()(int i, int j);
    int row_count();
    int column_count();
    void sort_features();
    bool sorted_features();
    void split(Dataset& v1, Dataset& v2, int column, int prop0, int prop1, double splitPoint);

    std::valarray<double> labels();
    std::valarray<double> feature(int index);
    std::valarray<double> row_features(int index);
    void set_row(int index, std::valarray<double>& prob);
    void add_dataset(Dataset& data);

    std::valarray<int> sort_info(int index);
    void copy_sort_info(int attr, std::valarray<int>& sort_info);
    void sort_feature(int attr, std::valarray<int>& sort_info);

private:
    Dataset( const Dataset& other );
    Dataset& operator=( const Dataset& rhs );

    std::shared_ptr<SharedData> m_sharedData;
    std::valarray<double>* m_dataFastPtr;

    int m_indent;
    int m_realRowNum;
    int m_rowNum;
    int m_columnNum;
    bool m_isSortIndBuilt;
};

class RandomForestClassifier;

class RandomTreeClassifier
{
    friend class boost::serialization::access;
public:
    RandomTreeClassifier();
    RandomTreeClassifier(int maxDepth, int numFeatures, bool debug);
    void fit(Dataset& data, int numClasses
                         , std::valarray<double>& vals
                         , std::valarray<double>& dists
                         , std::valarray<double>& props
                         , std::valarray<double>& splits
                         , std::valarray<bool>& allFeatureValuesEqual
                         , std::valarray<int>& sortInd
                         , std::valarray<double>& dist
                         , std::valarray<double>& currDist);
    void distribution(std::valarray<double>& instance, int depth, std::valarray<double>& prob);
    void distribution(std::valarray<double>& instance, std::valarray<int>& depth, std::valarray<double>& prob);

    void make_prediction(Dataset& data, std::valarray<int>& depth, Dataset& predictProb);

private:
    void build_tree(Dataset& data, std::valarray<double>& classProbs, std::valarray<int>& attIndicesWindow, int depth);
    double distribution(std::valarray<double>& props, std::valarray<double>& dists, int att, Dataset& data, bool& allFeatureValuesEqual);
    friend class RandomForestClassifier;

    std::valarray<RandomTreeClassifier*> m_trees;
    int m_attr;
    double m_splitPoint;
    std::valarray<double> m_prop;
    std::valarray<double> m_classDistribution;
    static const int m_minNum = 1;
    int m_numFeatures;
    int m_maxDepth;
    int m_currentDepth;
    int m_numClasses;
    int m_numFeaturesInBuildData;

    bool m_debug;
    std::valarray<double> *vals, *dists, *props, *splits, *dist, *currDist;
    std::valarray<bool> *allFeatureValuesEqual;
    std::valarray<int> *sortInd;

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/)
    {
        ar & m_debug;
        ar & m_attr;
        ar & m_splitPoint;
        ar & m_prop;
        ar & m_classDistribution;
        ar & m_numFeatures;
        ar & m_maxDepth;
        ar & m_currentDepth;
        ar & m_numClasses;

        if (m_attr >= 0)
        {
            if (m_trees.size() == 0)
            {
                m_trees.resize(2);
                m_trees[0] = new RandomTreeClassifier();
                m_trees[1] = new RandomTreeClassifier();
            }

            ar & (*m_trees[0]);
            ar & (*m_trees[1]);
        }
    }
};

class RandomForestClassifier
{
    friend class boost::serialization::access;
public:
    RandomForestClassifier();
    RandomForestClassifier(int numTrees, int maxDepth, int numFeatures, int randomSeed,
                           int numExecutionSlots, bool debug);

    void fit(Dataset& data);
    void distribution(std::valarray<double>& instance, int depth, std::valarray<double>& prob);
    void predict_proba(Dataset& data, std::valarray<int>& depth, Dataset& predictProb);
    int max_depth();
    int num_features();

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/)
    {
        ar & m_debug;
        ar & m_numTrees;
        ar & m_numFeatures;
        ar & m_randomSeed;
        ar & m_maxDepth;
        ar & m_numWorkers;
        ar & m_trees;
    }

private:
    void resample(Dataset& data, Dataset& result);

    bool m_debug;
    int m_numTrees;
    int m_numFeatures;
    int m_randomSeed;
    int m_maxDepth;
    int m_numWorkers;
    std::valarray<RandomTreeClassifier> m_trees;
};

#endif
