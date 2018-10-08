// #setup: import numpy ; a = numpy.array([ [i/10., i/10., i/20.] for i in range(80)], dtype=numpy.double)
// #run: hyantes(0., 0., 90., 90., 1., 100., 80, 80, a)

// #pythran export hyantes(float, float, float, float, float, float, int, int, float[][])
// import numpy as np
// def hyantes(xmin, ymin, xmax, ymax, step, range_, range_x, range_y, t):
//     X,Y = t.shape
//     pt = np.zeros((X,Y))
//     for i in range(X):
//         for j in range(Y):
//             for k in t:
//                 tmp = 6368.* np.arccos( np.cos(xmin+step*i)*np.cos( k[0] ) * np.cos((ymin+step*j)-k[1])+  np.sin(xmin+step*i)*np.sin(k[0]))
//                 if tmp < range_:
//                     pt[i,j]+=k[2] / (1+tmp)
//     return pt

namespace xt
{
	auto hyantes(double xmin, double ymin, double xmax, double ymax, double step,
				 double range_, double range_x, double range_y, const tensor<double, 2>& t)
	{
		tensor<double, 2> pt = xt::zeros_like(t);
		for (std::size_t i = 0; i < t.shape()[0]; ++i)
		{
			for (std::size_t j = 0; j < t.shape()[1]; ++j)
			{
				for (std::size_t k = 0; k < t.shape()[0]; ++k)
				{
					auto kv = xt::view(t, k);
					using namespace xt::math;
					double tmp = 6368. * acos(cos(xmin + step * i) * cos(kv(0))
									     * cos((ymin + step * j) - kv(1))+  sin(xmin + step * i)
									     * sin(kv(0)));
                	if(tmp < range_)
	                    pt(i, j) += kv(2) / (1 + tmp);
				}
			}
		}
		return pt;
	}
}
