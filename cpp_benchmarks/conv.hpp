namespace xt
{
    auto clamp_(std::ptrdiff_t i, std::ptrdiff_t offset, std::ptrdiff_t maxval) {
        auto j = std::max(std::ptrdiff_t(0), i + offset);
        return std::min(j, maxval);
    }

    auto reflect(std::ptrdiff_t pos, std::ptrdiff_t offset, std::ptrdiff_t bound) {
        auto idx = pos + offset;
        return std::min(2 * (bound - 1) - idx, std::max(idx, -idx));
    }

    auto conv(tensor<double, 2>& x, tensor<double, 2>& weights) {
        auto& sx = x.shape();
        auto& sw = weights.shape();
        auto result = xt::zeros_like(x);

        for (std::ptrdiff_t i = 0; i < sx[0]; ++i)
            for (std::ptrdiff_t j = 0; j < sx[1]; ++j)
                for (std::ptrdiff_t ii = 0; ii < sw[0]; ++ii)
                    for (std::ptrdiff_t jj = 0; jj < sw[1]; ++jj)
                    {
                        std::array<std::ptrdiff_t, 2> idx{clamp_(i, ii - std::floor(sw[0] / 2), sw[0]),
                                                          clamp_(j, jj - std::floor(sw[0] / 2), sw[0])};
                        result(i, j) += x[idx] * weights(ii, jj);
                    }
        return result;
    }
}