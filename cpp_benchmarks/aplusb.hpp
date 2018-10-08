namespace xt
{
	auto aplusb(const tensor<double, 1>& a, const tensor<double, 1>& b)
	{
		return tensor<double, 1>(a + b);
	}
}