#include "Shared.h"

#include <cassert>
#include <time.h>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>
#include <algorithm>

//using namespace std;

namespace Shared
{

struct ordering {
    bool operator ()(std::pair<int, double> const& a, std::pair<int, double> const& b) {return a.second < b.second;}
} Ordering;


void sort(double* x, int n, int* ind, std::pair<int, double>* order)
{
    for (int i = 0; i < n; i++)
        order[i] = std::make_pair(i, x[i]);
    std::sort(order, order + n, Ordering);
    for (int i = 0; i < n; i++)
        ind[i] = order[i].first;
}

int maxIndex(std::valarray<double>& x)
{
    size_t n = x.size();
    assert(n > 0);
    int res = 0;
    double val = x[0];
    for (size_t i=1; i<n; i++)
        if (x[i] > val) {val=x[i]; res=i;}
    return res;
}
int minIndex(std::valarray<double>& x)
{
    size_t n = x.size();
    assert(n > 0);
    int res = 0;
    double val = x[0];
    for (size_t i=1; i<n; i++)
        if (x[i] < val) {val=x[i]; res=i;}
    return res;
}

double max(std::valarray<double>& x)
{
    return x[maxIndex(x)];
}

double min(std::valarray<double>& x)
{
    return x[minIndex(x)];
}

double calculate_auc(const std::valarray<double>& prob, const std::valarray<double>& exact)
{
    std::valarray<double> tmp = prob;

    std::valarray<int> ind(tmp.size());
    std::valarray<std::pair<int, double> >  order(tmp.size());
    sort(&tmp[0], tmp.size(), &ind[0], &order[0]);
    double S0 = 0;
    int n = tmp.size(), n0=0, n1=0;

    int ii = 0;
    for (int i=0; i<n; i++)
    {
        if (exact[ind[i]] < -0.5)
            continue;

        if (exact[ind[i]] >= 0.5)
        {
            S0 += ii;
            n0++;
        }
        else
        {
            n1++;
            ii++;
        }
    }

    if ((n0 == 0)|| (n1 == 0))
        return 1;

    return S0/((double)n0*n1);
}

double calculate_accuracy(const std::valarray<double>& prob, const std::valarray<double>& exact)
{
    double result = 0;
    for (int i = 0; i < prob.size(); ++i)
        result += exact[i] == round(prob[i]);
    result = result / prob.size();
    return result;
}

}
