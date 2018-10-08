// #setup: import numpy as np, random; np.random.seed(0); s=np.random.randn(2**16)+np.random.randn(2**16)*1.j ; sc=np.random.choice(s, 32)
// #run: make_decision(s, sc)
// #from: https://github.com/serge-sans-paille/pythran/issues/801

// import numpy as np
// #pythran export make_decision(complex128[], complex128[])
// def make_decision(E, symbols):
//     L = E.shape[0]
//     syms_out = np.zeros(L, dtype=E.dtype)
//     for i in range(L):
//         im = np.argmin(abs(E[i]-symbols)**2)
//         syms_out[i] = symbols[im]
//     return syms_out

namespace xt
{
	template <class E>
	auto abssqr(E&& e)
	{
		auto lambda = [](auto x) -> decltype(x.imag() * x.imag()) { return x.imag() * x.imag() + x.real() * x.real(); };
		return make_lambda_xfunction(std::move(lambda), std::forward<E>(e));
	}

	template <class T>
	void pt(T)
	{
		std::cout << __PRETTY_FUNCTION__ << std::endl;
	}

	using complex = std::complex<double>;
	auto make_decision(const tensor<complex, 1>& E, const tensor<complex, 1>& symbols)
	{
		auto L = E.shape()[0];
		tensor<complex, 1> syms_out = xt::ones<complex>({L});

		for (std::size_t i = 0; i < L; ++i)
		{
			auto f = abssqr(E(i) - symbols);
			double val = 1e100;
			std::size_t idx = 0;
			alignas(32) double vals[4];
			for (std::size_t j = 0; j < symbols.shape()[0] / 4; ++j) 
			{
				auto b = f.load_simd<xsimd::unaligned_mode>(j * 4);
				b.store_aligned(vals);
				for (int k = 0; k < 4; ++k)
				{
					double tmp = vals[k];				
					if (tmp < val)
					{
						tmp = val;
						idx = j + k;
					}
				}
			}
			// std::size_t im = static_cast<std::size_t>(xt::argmin(abssqr(E(i) - symbols))());
			syms_out(i) = symbols(idx);
		}

		return syms_out;
	}
}