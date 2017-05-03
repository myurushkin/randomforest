#ifndef RANDOM_FOREST_SHARED_H
#define RANDOM_FOREST_SHARED_H

#include <string>
#include <time.h>
#include <valarray>

namespace Shared
{

void sort(double* x, int n, int* ind, std::pair<int, double>* order);

int maxIndex(std::valarray<double>& x);
int minIndex(std::valarray<double>& x);

double max(std::valarray<double>& x);
double min(std::valarray<double>& x);
double calculate_auc(const std::valarray<double>& prob, const std::valarray<double>& exact);
double calculate_accuracy(const std::valarray<double>& prob, const std::valarray<double>& exact);

}

#endif
