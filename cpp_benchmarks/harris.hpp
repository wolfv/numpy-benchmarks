namespace xt
{
    tensor<double, 2> harris(const tensor<double, 2>& I)
    {
        using namespace xt::placeholders;

        std::size_t m = I.shape()[0], n = I.shape()[1];

        tensor<double, 2> dx = view(view(I, range(1, _), all()) - view(I, range(0, m - 1), all()), all(), range(1, _));
        tensor<double, 2> dy = view(view(I, all(), range(1, _)) - view(I, all(), range(0, n - 1)), range(1, _), all());

        tensor<double, 2> A = dx * dx;
        tensor<double, 2> B = dy * dy;
        tensor<double, 2> C = dx * dy;

        auto tr = A + B;
        auto det = A * B - square(C);
        double k = 0.05;
        return det - k * square(tr);
    }
}