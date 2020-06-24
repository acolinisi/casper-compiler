void mat_abs(double *M_buf, double *M, int offset,
		int n, int m, int s_n, int s_m) {
	printf("%d, %d %d, %d %d\n", offset, n, m, s_n, s_m);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			float v = M[i * s_n + j * s_m];
			if (v < 0)
				M[i * s_n + j * s_m] = -v;
		}
	}
}

void mat_double(double *M_buf, double *M, int offset,
		int n, int m, int s_n, int s_m) {
	printf("%d, %d %d, %d %d\n", offset, n, m, s_n, s_m);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
				M[i * s_n + j * s_m] *= 2;
		}
	}
}

void mat_tripple(double *M_buf, double *M, int offset,
		int n, int m, int s_n, int s_m) {
	printf("%d, %d %d, %d %d\n", offset, n, m, s_n, s_m);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
				M[i * s_n + j * s_m] *= 3;
		}
	}
}

void mat_invert(double *M_buf, double *M, int offset,
		int n, int m, int s_n, int s_m) {
	printf("%d, %d %d, %d %d\n", offset, n, m, s_n, s_m);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
				M[i * s_n + j * s_m] *= -1;
		}
	}
}
