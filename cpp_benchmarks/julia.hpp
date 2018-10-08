namespace xt
{
    inline auto kernel(double zr, double zi, double cr, double ci, double lim, double cutoff)
    {
        uint32_t count = 0.;
        while (((zr*zr + zi*zi) < (lim*lim)) and count < cutoff)
        {
            zr = zr * zr - zi * zi + cr;
            zi = 2. * zr * zi + ci;
            count += 1;
        }
        return count;
    }

    auto julia(double cr, double ci, std::size_t N, double bound=1.5, double lim=1000., double cutoff=1e6)
    {
        auto julia = xt::empty<uint32_t>({N, N});
        // grid_x = np.linspace(-bound, bound, N)
        tensor<double, 1> grid_x = xt::linspace<double>(-bound, bound, N);

        for (std::size_t i = 0; i < N; ++i)
        {
            auto& x = grid_x(i);
            for (std::size_t j = 0; j < N; ++j)
            {
                auto& y = grid_x(j);
                julia(i, j) = kernel(x, y, cr, ci, lim, cutoff);
            }
        }
        return julia;
    }
}
