void mat_norm(double *M_buf, double *M, int offset,
		int n, int m, int s_n, int s_m) {

	printf("%d, %d %d, %d %d\n", offset, n, m, s_n, s_m);

	double sum_sq = 0;

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			double v = M[i * s_n + j * s_m];
			sum_sq += v * v;
		}
	}

	for (int i = 0; i < n; ++i)
		for (int j = 0; j < m; ++j)
			M[i * s_n + j * s_m] /= sum_sq;
}
