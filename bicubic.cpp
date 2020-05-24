// #include <boost/range/combine.hpp>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <ratio>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>

using namespace xt;

int main(int argc, char **argv)
{
    static const int n = 1000000;

    auto a = eval(random::rand<double>({406, 406, 4, 4}));

    xtensor_fixed<double, xshape<2>> fx
        {1.0, 1.0};

    xtensor_fixed<double, xshape<2>> x0
        {3.0, 3.0};

    xtensor_fixed<double, xshape<2>> xlength
        {400.0, 400.0};

    for (int run = 0; run < 2; run ++)
    {
        xarray<double> x = eval(random::rand<double>({n, 2}, 0, 400));
        double y;

        auto t1 = std::chrono::steady_clock::now();

        auto x_clipped = clip(fma(x, fx, x0), 0, xlength - 1);
        auto ix_double = floor(x_clipped);
        xarray<double> st = x_clipped - ix_double;
        xarray<int> ix = eval(cast<int>(ix_double));

        auto ix_iter = axis_begin(ix, 0);
        auto ix_end = axis_end(ix, 0);
        auto xx_iter = axis_begin(st, 0);
        auto xx_end = axis_begin(st, 0);
        while (ix_iter != ix_end)
        {
            auto ii = *ix_iter++;
            auto xx = *xx_iter++;
            auto s = xx(0);
            auto t = xx(1);
            auto aa = view(a, ii(0), ii(1), all(), all());
            auto bb = fma(t, fma(t, fma(t, row(aa, 0), row(aa, 1)), row(aa, 2)), row(aa, 3));
            auto cc = fma(s, fma(s, fma(s, bb(0), bb(1)), bb(2)), bb(3));
            y = cc;
        }

        auto t2 = std::chrono::steady_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
        std::cout << time_span.count() << std::endl;
    }

    return 0;
}
