#include "RandomForestClassifier.h"
#include <iostream>
#include "Shared.h"
#include <assert.h>
#include <chrono>
#include <omp.h>

using namespace std;

Dataset::Dataset()
{
    m_indent = 0;
    m_rowNum = 0;
    m_realRowNum = 0;
    m_columnNum = 0;
    m_isSortIndBuilt = false;
}
Dataset::Dataset(int rowNum, int columnNum):m_rowNum(rowNum), m_columnNum(columnNum)
{
    m_realRowNum = m_rowNum;
    m_indent = 0;
    m_sharedData = std::shared_ptr<SharedData>(new SharedData());
    m_sharedData->data.resize(rowNum*columnNum, 0);
    m_isSortIndBuilt = false;
    m_sharedData->indBuffer.resize(rowNum);
    m_sharedData->orderBuffer.resize(rowNum);
    m_dataFastPtr = &(m_sharedData->data);
}

void Dataset::set_data(int rowNum, int columnNum, const std::valarray<double>& data)
{
    assert(data.size() == rowNum * columnNum);
    m_rowNum = rowNum;
    m_realRowNum = m_rowNum;
    m_columnNum = columnNum;
    m_indent = 0;
    m_sharedData = std::shared_ptr<SharedData>(new SharedData());
    m_sharedData->data = data;
    m_isSortIndBuilt = false;
    m_sharedData->indBuffer.resize(rowNum);
    m_sharedData->orderBuffer.resize(rowNum);
    m_dataFastPtr = &(m_sharedData->data);
}

void Dataset::set_newSize(int rowNum, int columnNum)
{
    m_rowNum = rowNum;
    m_realRowNum = m_rowNum;
    m_columnNum = columnNum;
    m_indent = 0;
    m_sharedData = std::shared_ptr<SharedData>(new SharedData());
    m_sharedData->data.resize(rowNum*columnNum, 0);
    m_isSortIndBuilt = false;
    m_sharedData->indBuffer.resize(rowNum);
    m_sharedData->orderBuffer.resize(rowNum);
    m_dataFastPtr = &(m_sharedData->data);
}
void Dataset::prepare()
{
    m_indent = 0;
    m_isSortIndBuilt = false;
}

double& Dataset::operator()(int i, int j)
{
    return (*m_dataFastPtr)[m_indent + j*m_realRowNum+i];
}

int Dataset::row_count()
{
    return m_rowNum;
}

int Dataset::column_count()
{
    return m_columnNum;
}

void Dataset::sort_features()
{
    assert(m_indent == 0 && m_realRowNum == m_rowNum);
    assert(m_isSortIndBuilt == false);
    m_sharedData->sortInd.resize(m_rowNum*(m_columnNum-1));
    for (int j=0; j<m_columnNum-1; j++)
    {
        Shared::sort(&m_sharedData->data[m_indent + j*m_realRowNum], m_rowNum,
                &m_sharedData->sortInd[m_indent + j*m_realRowNum],
                &m_sharedData->orderBuffer[0]);
    }
    m_isSortIndBuilt = true;
}

bool Dataset::sorted_features()
{
    return m_isSortIndBuilt;
}


std::valarray<double> Dataset::labels()
{
    int m = row_count();
    int n = column_count();
    return m_sharedData->data[slice(m_indent + m_realRowNum*(n-1),m,1)];
}

std::valarray<double> Dataset::feature(int index)
{
    int m = row_count();
    return m_sharedData->data[slice(m_indent + m_realRowNum*index,m,1)];
}

std::valarray<double> Dataset::row_features(int index)
{
    int m = row_count();
    int n = column_count() - 1;
    return m_sharedData->data[slice(m_indent + index, n, m_realRowNum)];
}

void Dataset::set_row(int index, valarray<double>& prob)
{
    int m = row_count();
    int n = column_count();
    assert(n == prob.size());
    m_sharedData->data[slice(m_indent + index, n, m_realRowNum)] = prob;
}

void Dataset::add_dataset(Dataset& data)
{
    assert(m_indent == 0 && data.m_indent == 0);
    m_sharedData->data += data.m_sharedData->data;
}

std::valarray<int> Dataset::sort_info(int index)
{
    int m = row_count();
    return m_sharedData->sortInd[slice(m_indent + m_realRowNum*index,m,1)];
}

void Dataset::copy_sort_info(int attr, std::valarray<int>& sortInd)
{
    int m = row_count();

    auto& source = m_sharedData->sortInd;
    for (int i =0; i < m; i++)
        sortInd[i] = source[m_indent + m_realRowNum*attr + i];
}
void Dataset::sort_feature(int attr, valarray<int>& sortInd)
{
    assert(sorted_features() == false);
    int nrows = row_count();
    Shared::sort(&m_sharedData->data[m_indent + m_realRowNum*attr], nrows, &sortInd[0], &m_sharedData->orderBuffer[0]);
}

RandomTreeClassifier::RandomTreeClassifier()
{
    m_debug = false;
    m_attr = -1000000000;
    m_splitPoint = -1e15;
    m_numFeatures = -1000000000;
    m_maxDepth = -1000000000;
    m_currentDepth = 0;
    m_numFeatures = -1;
}

RandomTreeClassifier::RandomTreeClassifier(int maxDepth, int numFeatures, bool debug)
{
    m_debug = debug;
    m_attr = -1000000000;
    m_splitPoint = -1e15;
    m_numFeatures = numFeatures;
    m_maxDepth = maxDepth;
    m_currentDepth = 0;
    m_numFeatures = -1;
}

void RandomTreeClassifier::fit(Dataset& data, int numClasses, std::valarray<double>& vals
                                           , std::valarray<double>& dists
                                           , std::valarray<double>& props
                                           , std::valarray<double>& splits
                                           , std::valarray<bool>& allFeatureValuesEqual
                                           , std::valarray<int>& sortInd
                                           , std::valarray<double>& dist
                                           , std::valarray<double>& currDist)
{
    this->vals = &vals;
    this->dists = &dists;
    this->props = &props;
    this->splits = &splits;
    this->allFeatureValuesEqual = &allFeatureValuesEqual;
    this->sortInd = &sortInd;
    this->dist = &dist;
    this->currDist = &currDist;


    m_numClasses = numClasses;
    int nrows = data.row_count(), ncols = data.column_count();
    m_numFeatures = ncols-1;
    assert((m_numFeatures > 0) && (m_numFeatures <= ncols-1));
    
    // Create the attribute indices window
    valarray<int> attIndicesWindow(ncols-1);
    for (int i = 0; i < (int)attIndicesWindow.size(); i++)
        attIndicesWindow[i] = i;

    // Compute initial class counts
    valarray<double> classProbs(numClasses);
    for (int i = 0; i < nrows; i++)
        classProbs[(int) data(i,ncols-1)]++;

    build_tree(data, classProbs, attIndicesWindow, 0);
}

inline double log_func(double num)
{
    return (num < 1e-6) ? 0 : num * log(num);
}

static const double log2const = log(2.0);

inline double entropyOverColumns(valarray<double>& matrix, int nrows, int ncols)
{
    double returnValue = 0, sumForColumn, total = 0;

    for (int j = 0; j < ncols; j++)
    {
        sumForColumn = 0;
        for (int i = 0; i < nrows; i++) sumForColumn += matrix[i+nrows*j];
        returnValue = returnValue - log_func(sumForColumn);
        total += sumForColumn;
    }
    if (abs(total)<1e-6) return 0;
    return (returnValue + log_func(total)) / (total * log2const);

}

inline double entropyConditionedOnRows(valarray<double>& matrix, int nrows, int ncols)
{
    double returnValue = 0, sumForRow, total = 0;

    for (int i = 0; i < nrows; i++) {
        sumForRow = 0;
        for (int j = 0; j < ncols; j++)
        {
            returnValue = returnValue + log_func(matrix[i+nrows*j]);
            sumForRow += matrix[i+nrows*j];
        }
        returnValue = returnValue - log_func(sumForRow);
        total += sumForRow;
    }
    if (abs(total)<1e-6) return 0;
    return -returnValue / (total * log2const);

}

inline double gain(valarray<double>& dist, int nrows, int ncols, double priorVal) 
{
    return priorVal - entropyConditionedOnRows(dist, nrows, ncols);
}

double RandomTreeClassifier::distribution(valarray<double>& props, valarray<double>& dists, int att, Dataset& data, bool& allFeatureValuesEqual)
{
    assert((att>=0) && (att < data.column_count()-1));
    allFeatureValuesEqual = false;
    int nrows = data.row_count(), ncols = data.column_count(), ncols1 = ncols-1;

    valarray<double>& dist = *this->dist;
    valarray<double>& currDist = *this->currDist;
    dist = 0.0;
    currDist = 0.0;


    valarray<int>& sortInd = *this->sortInd;
    if (data.sorted_features())
        data.copy_sort_info(att, sortInd);
    else
        data.sort_feature(att, sortInd);

    for (int i = 0; i < nrows; i++)
        currDist[1+2*(int)data(i,ncols1)]++;

    double priorVal = entropyOverColumns(currDist, 2, m_numClasses);
    dist = currDist;

    double currSplit = data(sortInd[0], att);
    double splitPoint = currSplit >=0 ? -1 : 2*currSplit;
    double currVal, bestVal = -1e16;
    for (int i = 0; i < nrows; i++)
    {
        if (data(sortInd[i],att) > currSplit)
        {
            currVal = gain(currDist, 2, m_numClasses, priorVal);
            if (currVal > bestVal)
            {
                bestVal = currVal;
                splitPoint = data(sortInd[i],att);
                dist = currDist;
            }
        }
        currSplit = data(sortInd[i],att);
        currDist[0+2*(int)data(sortInd[i],ncols1)]++;
        currDist[1+2*(int)data(sortInd[i],ncols1)]--;
    }

    for (int k = 0; k < 2; k++)
    {
        double s = 0;
        for (int i=0; i < m_numClasses; i++) s += dist[k+2*i];
        props[att+k*ncols1] = s;
    }
    double s = props[att]+props[att+ncols1];
    if (abs(s)<1e-6)
    {
        props[att+ncols1*0] = nrows/2;
        props[att+ncols1*1] = nrows-props[att+ncols1*0];
    }
    dists[slice(att, 2*m_numClasses, ncols1)] = dist;
    if (data(sortInd[0], att) >= splitPoint)
        allFeatureValuesEqual = true;
    if (data(sortInd[0], att) == data(sortInd[nrows-1], att))
        allFeatureValuesEqual = true;
    return splitPoint;
}

void Dataset::split(Dataset& v0, Dataset& v1, int column, int prop0, int prop1, double splitPoint)
{
    int nrows = row_count();
    int ncols = column_count();

    assert(prop0 + prop1 == nrows);

    v0.m_sharedData = m_sharedData;
    v0.m_indent = m_indent;
    v0.m_rowNum = prop0;
    v0.m_realRowNum = m_realRowNum;
    v0.m_columnNum = m_columnNum;
    v0.m_isSortIndBuilt = false;
    v0.m_dataFastPtr = &(v0.m_sharedData->data);

    v1.m_sharedData = m_sharedData;
    v1.m_indent = m_indent + prop0;
    v1.m_rowNum = prop1;
    v1.m_realRowNum = m_realRowNum;
    v1.m_columnNum = m_columnNum;
    v1.m_isSortIndBuilt = false;
    v1.m_dataFastPtr = &(v1.m_sharedData->data);

    int i0 = 0;
    int i1 = 0;

    while (true)
    {
        while (i0 < v0.row_count() && v0(i0, column) < splitPoint)
        {
            i0 += 1;
            continue;
        }

        while (i1 < v1.row_count() && v1(i1, column) >= splitPoint)
        {
            i1 += 1;
            continue;
        }

        bool check1 = i0 < v0.row_count();
        bool check2 = i1 < v1.row_count();
        assert(check1 == check2);
        if (check1 == false)
            break;

        for (int i = 0; i < ncols; i++)
            ::swap(v0(i0, i), v1(i1, i));

        i0 += 1;
        i1 += 1;
    }
}

void RandomTreeClassifier::build_tree(Dataset& data, valarray<double>& classProbs, valarray<int>& attIndicesWindow, int depth)
{
    valarray<double>& vals = *this->vals;
    valarray<double>& dists = *this->dists;
    valarray<double>& props = *this->props;
    valarray<double>& splits = *this->splits;
    valarray<bool>& allFeatureValuesEqual = *this->allFeatureValuesEqual;

    int nrows = data.row_count(), ncols = data.column_count(), ncols1 = ncols-1;
    if (nrows == 0)
    {
        m_attr = -1;
        m_classDistribution.resize(classProbs.size(), 0);
        m_prop.resize(classProbs.size(), 0);
        return;
    }

    m_classDistribution.resize(classProbs.size());
    m_classDistribution = classProbs;
    if ((m_classDistribution.sum() < 2*m_minNum)
            || (abs(Shared::max(m_classDistribution) - m_classDistribution.sum()) < 1e-6)
            || ((m_maxDepth > 0) && (depth >= m_maxDepth)))
    {
        m_attr = -1;
        m_prop.resize(0);
        return;
    }

    vals = 0.0;
    dists = 0.0;
    props = 0.0;
    splits = 0.0;
    allFeatureValuesEqual = false;

    int attIndex = 0;
    int windowSize = attIndicesWindow.size();
    int k = m_numFeatures;
    bool gainFound = false;
    while ((windowSize > 0) && ((k-- > 0) || !gainFound))
    {
        int chosenIndex = m_debug ? ((int)floor(abs(sin(k*1.0))*100)) % windowSize : rand() % windowSize;
        attIndex = attIndicesWindow[chosenIndex];

        // shift chosen attIndex out of window
        attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
        attIndicesWindow[windowSize - 1] = attIndex;
        windowSize--;

        splits[attIndex] = distribution(props, dists, attIndex, data, allFeatureValuesEqual[attIndex]);
        slice sl(attIndex, 2*m_numClasses, ncols1);
        valarray<double> dist(2*m_numClasses);
        dist = dists[sl];
        vals[attIndex] = gain(dist, 2, m_numClasses, entropyOverColumns(dist, 2, m_numClasses));

        if (vals[attIndex] > 1e-6) gainFound = true;
    }

    // Find best attribute
    if (m_numFeatures>1) m_attr = Shared::maxIndex(vals);
    else  m_attr = attIndex;
    assert(allFeatureValuesEqual[m_attr] == false || vals[m_attr]==0);
    valarray<double> distribution1(2*m_numClasses);
    distribution1 = dists[slice(m_attr, 2*m_numClasses, ncols1)];

    if ((vals[m_attr]>1e-6) || (m_numFeatures==1))
    {
        m_splitPoint = splits[m_attr];
        m_prop.resize(2);
        m_prop = props[slice(m_attr,2,ncols1)];
        Dataset subset0, subset1;
        valarray<double> dist0(m_numClasses);
        dist0 = distribution1[slice(0,m_numClasses,2)];
        valarray<double> dist1(m_numClasses);
        dist1 = distribution1[slice(1,m_numClasses,2)];
        data.split(subset0, subset1, m_attr, m_prop[0], m_prop[1], m_splitPoint);
        m_trees.resize(2);
        m_trees[0] = new RandomTreeClassifier(*this);
        m_trees[0]->m_currentDepth = m_currentDepth+1;
        m_trees[0]->build_tree(subset0, dist0, attIndicesWindow, depth+1);

        m_trees[1] = new RandomTreeClassifier(*this);
        m_trees[1]->m_currentDepth = m_currentDepth+1;
        m_trees[1]->build_tree(subset1, dist1, attIndicesWindow, depth+1);
    }
    else
    {
        // Make leaf
        m_attr = -1;
    }
}


void RandomTreeClassifier::distribution(valarray<double>& instance, int depth, valarray<double>& prob)
{
    if ((depth != -1)&&(depth <= m_currentDepth))
    {
        if (m_classDistribution.size() == 0)
        {
            assert(m_classDistribution.size() > 0);
            prob.resize(0);
            return;
        }
        prob.resize(m_classDistribution.size());
        prob = m_classDistribution;
        prob /= prob.sum();
        return;
    }

    if (m_attr > -1)
    {
        if (instance[m_attr] < m_splitPoint)
            m_trees[0]->distribution(instance, depth, prob);
        else
            m_trees[1]->distribution(instance, depth, prob);
        return;
    }
    else
    {
        if (m_classDistribution.size() == 0)
        {
            assert(m_classDistribution.size() > 0);
            prob.resize(0);
            return;
        }
        prob.resize(m_classDistribution.size());
        prob = m_classDistribution;
        prob /= prob.sum();
        return;
    }
}

void RandomTreeClassifier::distribution(valarray<double>& instance, valarray<int>& depth,
                                                   valarray<double>& prob)
{
    assert(instance.size() == m_numFeatures);
    int depthNum = depth.size();
    if (prob.size() != depthNum*m_numClasses) prob.resize(depthNum*m_numClasses, 0);
    bool flagIsEqual = false;
    int i=0;
    for (; (i<depthNum) && (depth[i]<= m_currentDepth); i++)
        if (m_currentDepth == depth[i]) {flagIsEqual=true; break;}
    if (flagIsEqual)
    {
        assert(m_classDistribution.size() > 0);
        valarray<double> p(m_classDistribution.size());
        p = m_classDistribution;
        p /= p.sum();
        prob[slice(i*m_numClasses, m_numClasses, 1)] = p;
        if (i == depthNum-1)  return;
    }

    if (m_attr > -1)
    {
        if (instance[m_attr] < m_splitPoint)
            m_trees[0]->distribution(instance, depth, prob);
        else
            m_trees[1]->distribution(instance, depth, prob);
    }
    else
    {
        assert(m_classDistribution.size() > 0);
        valarray<double> p(m_classDistribution.size());
        p = m_classDistribution;
        p /= p.sum();
        for (int i1 = i; i1 < depthNum; i1++)
            prob[slice(i1*m_numClasses, m_numClasses, 1)] = p;
    }
}

void RandomTreeClassifier::make_prediction(Dataset& data, std::valarray<int>& depth, Dataset& predictProb)
{
    int nrows = data.row_count(), ncols = m_numFeatures;
    assert((ncols == data.column_count()) || (ncols == data.column_count()-1));
    for (int i=0; i < nrows; i++)
    {
        valarray<double> instance(ncols);
        instance = data.row_features(i);
        valarray<double> prob;
        distribution(instance, depth, prob);
        predictProb.set_row(i, prob);
    }
}

RandomForestClassifier::RandomForestClassifier()
{
    m_debug = false;
}
RandomForestClassifier::RandomForestClassifier(int numTrees, int maxDepth, int numFeatures,
                                               int randomSeed, int numWorkers, bool debug)
{
    numTrees = (numTrees / numWorkers + 1) * numWorkers;
    m_debug = debug;
    m_numTrees = numTrees;
    RandomTreeClassifier exampleTree(maxDepth, numFeatures, debug);
    m_trees.resize(numTrees, exampleTree);
    m_numFeatures = numFeatures;
    m_numWorkers = numWorkers;
    m_randomSeed = randomSeed;
    if (m_debug) m_numWorkers = 1;
    m_numFeatures = -1;
    m_maxDepth = maxDepth;
}

int calcUniqueValueNum(valarray<double>& data)
{
    valarray<int> ind(data.size());
    std::vector<std::pair<int, double> > order(data.size());
    Shared::sort(&data[0], data.size(), &ind[0], &order[0]);
    int res = 0;
    for (size_t i = 0; i < data.size()-1; i++)
        if (data[ind[i]] != data[ind[i+1]]) res++;
    return res+1;
}

void RandomForestClassifier::fit(Dataset& data)
{
    int nrows = data.row_count(), ncols = data.column_count();
    m_numFeatures = ncols-1;
    if ((m_numTrees*m_numFeatures*m_maxDepth > ncols-1) && !data.sorted_features())
        data.sort_features();
    srand(m_randomSeed);
    omp_set_num_threads(m_numWorkers);
    valarray<double> label(nrows);
    label = data.labels();
    int numClasses = calcUniqueValueNum(label);
    assert(numClasses == 2);

#pragma omp parallel
    {
        int numThreads = omp_get_num_threads();
        int jobSize = m_numTrees/numThreads;

        std::valarray<double> vals, dists, props, splits, dist, currDist;
        std::valarray<bool> allFeatureValuesEqual;
        std::valarray<int> sortInd;

        int ncols1 = ncols-1;
        vals.resize(ncols - 1);
        dists.resize(ncols1*2*numClasses);
        props.resize(ncols1*2);
        splits.resize(ncols1);
        allFeatureValuesEqual.resize(ncols1);
        sortInd.resize(nrows);
        dist.resize(2*numClasses);
        currDist.resize(2*numClasses);
        Dataset newData(nrows, ncols);

        int p = omp_get_thread_num();
        for (int i = p*jobSize; i < (p+1)*jobSize; i++)
        {
            newData.prepare();
            resample(data, newData);
            m_trees[i].fit(newData, numClasses, vals, dists, props, splits, allFeatureValuesEqual, sortInd, dist, currDist);
        }
    }
}

void RandomForestClassifier::distribution(valarray<double>& instance, int depth, valarray<double>& prob)
{
    if (m_trees.size() == 0) {prob.resize(0); return;}
    valarray<double> newProb;
    prob.resize(m_trees[0].m_numClasses, 0);
    for (int i = 0; i < m_numTrees; i++)
    {
        m_trees[i].distribution(instance, depth, newProb);
        prob += newProb;
    }
    prob /= prob.sum();
}

void RandomForestClassifier::predict_proba(Dataset& data, std::valarray<int>& depth, Dataset& predictProb)
{
    int numClasses = m_trees[0].m_numClasses;
    int nrows = data.row_count(), ncols = data.column_count();
    assert((ncols == m_numFeatures) || (ncols == m_numFeatures+1));
    predictProb.set_newSize(nrows, depth.size()*numClasses);
    Dataset prob(nrows, depth.size()*numClasses);
    for (int i=0; i<m_numTrees; i++)
    {
        m_trees[i].make_prediction(data, depth, prob);
        predictProb.add_dataset(prob);
    }

    for (int i=0; i<nrows; i++)
    {
        for (size_t d=0; d<depth.size(); d++)
        {
            double s = 0;
            for (int j=0; j<numClasses; j++)
                s += predictProb(i, d*numClasses+j);
            for (int j=0; j<numClasses; j++)
                predictProb(i, d*numClasses+j) /= s;
        }
    }
}

int RandomForestClassifier::max_depth()
{
    return m_maxDepth;
}
int RandomForestClassifier::num_features()
{
    return m_numFeatures;
}

void RandomForestClassifier::resample(Dataset& data, Dataset& newData)
{
    int nrows = data.row_count(), ncols = data.column_count();
    if (nrows*ncols == 0) return;
    valarray<double> probabilities(nrows);
    double sumProbs = 0, sumOfWeights = nrows;
    for (int i = 0; i < nrows; i++)
    {
        if (m_debug) sumProbs += 1;
        else sumProbs += (double)rand()/RAND_MAX;
        probabilities[i] = sumProbs;
    }
    probabilities /= sumProbs/sumOfWeights;
    probabilities[nrows - 1] = sumOfWeights;
    int k = 0; int l = 0;
    sumProbs = 0;
    while ((k < nrows && (l < nrows)))
    {
        sumProbs += 1;
        while ((k < nrows) && (probabilities[k] <= sumProbs))
        {
            for (int j=0; j<ncols; j++) newData(k,j) = data(l,j);
            k++;
        }
        l++;
    }
    assert((k == nrows) && (l == nrows));
}

