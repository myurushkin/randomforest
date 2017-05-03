#include "../include/RandomForestClassifier.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <omp.h>
#include "../include/Shared.h"

using namespace std;

void loadDataset(std::string fileName, Dataset& features, bool containsLabels)
{
    int objectCount, featureCount;

    ifstream in(fileName);
    if (in.is_open() == false)
        throw std::runtime_error("couldn't open input file");

    in >> objectCount >> featureCount;
    valarray<double> values(objectCount * featureCount);
    for (int i = 0; i < objectCount; ++i) {
        for (int j = 0; j < featureCount; ++j)
            in >> values[j*objectCount + i];
    }

    features.set_data(objectCount, featureCount, values);
}


int main(int argc, char* argv[]) {
    string mode;
    string inputPath, outputPath;
    string modelPath;

    int numTrees = -1;
    int maxDepth = -1;
    int numFeatures = -1;
    int randomSeed = 1234;
    int threadNumber = -1;
    for (int i = 1; i < argc - 1; ++i)
    {
        if (strcmp(argv[i], "--mode") == 0)
            mode = argv[i+1];
        if (strcmp(argv[i], "--input") == 0)
            inputPath = argv[i+1];
        if (strcmp(argv[i], "--output") == 0)
            outputPath = argv[i+1];
        if (strcmp(argv[i], "--model") == 0)
            modelPath = argv[i+1];
        if (strcmp(argv[i], "--num-trees") == 0)
            numTrees = atoi(argv[i+1]);
        if (strcmp(argv[i], "--max-depth") == 0)
            maxDepth = atoi(argv[i+1]);
        if (strcmp(argv[i], "--num-features") == 0)
            numFeatures = atoi(argv[i+1]);
        if (strcmp(argv[i], "--num-threads") == 0)
            threadNumber = atoi(argv[i+1]);
    }

    if (modelPath.empty() || inputPath.empty() || modelPath.empty())
    {
        cout << "input parameters are invalid.";
        return 1;
    }


    Dataset data;
    try {
        loadDataset(inputPath, data, mode != "validate");
    }
    catch(...)
    {
        cout << "couldn't read input file";
        return 2;
    }

    if (mode == "train")
    {
        cout << "training..." << std::endl;
        double start = omp_get_wtime();

        RandomForestClassifier model(numTrees, maxDepth, numFeatures, randomSeed, threadNumber, false);
        model.fit(data);
        std::ofstream out(modelPath);
        boost::archive::text_oarchive oa(out);
        oa << model;

        double end = omp_get_wtime();
        std::cout << "Elapsed: " << end - start << std::endl;
    }

    if (mode == "test")
    {
        cout << "testing..." << std::endl;
        RandomForestClassifier model;
        std::ifstream in(modelPath);
        boost::archive::text_iarchive ia(in);
        ia >> model;

        valarray<int> depth(1);
        depth[0] = model.max_depth();

        Dataset predictProb;
        model.predict_proba(data, depth, predictProb);
        std::cout << "AUC: " << Shared::calculate_auc(predictProb.labels(), data.labels()) << std::endl;
        std::cout << "ACC: " << Shared::calculate_accuracy(predictProb.labels(), data.labels()) << std::endl;

        int count = predictProb.labels().size();
        auto labels = data.labels();
        auto probs = predictProb.labels();

        ofstream fout(outputPath);
        for (int i = 0; i < count; ++i)
            fout << labels[i] << " " << probs[i] << std::endl;
    }

    if (mode == "validate")
    {
        cout << "validating..." << std::endl;
        RandomForestClassifier model;
        std::ifstream in(modelPath);
        boost::archive::text_iarchive ia(in);
        ia >> model;

        valarray<int> depth(1);
        depth[0] = model.max_depth();

        Dataset predictProb;
        model.predict_proba(data, depth, predictProb);

        int count = predictProb.labels().size();
        auto probs = predictProb.labels();

        ofstream fout(outputPath);
        for (int i = 0; i < count; ++i)
            fout << probs[i] << std::endl;
    }


    return 0;
}
