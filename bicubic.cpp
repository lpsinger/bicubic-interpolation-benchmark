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
using namespace std;
using namespace std::chrono;

int main(int argc, char **argv)
{
    static const int n = 100000;

    auto a = eval(random::rand<double>({406 * 406, 4, 4}));

    xtensor_fixed<double, xshape<2>> fx
        {1.0, 1.0};

    xtensor_fixed<double, xshape<2>> x0
        {3.0, 3.0};

    xtensor_fixed<double, xshape<2>> xlength
        {400.0, 400.0};

    for (int run = 0; run < 2; run ++)
    {
        xtensor_fixed<double, xshape<n, 2>> x = eval(random::rand<double>({n, 2}, 0, 400));
        double y;

        auto t1 = steady_clock::now();

        auto&& x_clipped = eval(clip(fma(x, fx, x0), 0, xlength - 1));
        auto&& ix_double = eval(floor(x_clipped));
        auto&& st = eval(x_clipped - ix_double);
        auto&& ix = eval(cast<int>(fma(406.0, col(ix_double, 0), col(ix_double, 1))));

        for (int i = 0; i < n; i ++)
        {
            auto s = st(i, 0);
            auto t = st(i, 1);
            auto&& aa = view(a, ix(i), all(), all());
            auto&& bb = fma(t, fma(t, fma(t, row(aa, 0), row(aa, 1)), row(aa, 2)), row(aa, 3));
            y = fma(s, fma(s, fma(s, bb(0), bb(1)), bb(2)), bb(3));
        }

        auto t2 = steady_clock::now();
        auto time_span = duration_cast<duration<double> >(t2 - t1);
        cout << time_span.count() << endl;
    }

    return 0;
}
