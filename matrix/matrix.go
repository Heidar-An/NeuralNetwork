package matrix

type Matrix [][]float64

func New(row, col int) Matrix {
	mat := make([][]float64, row)
	for i := range mat {
		mat[i] = make([]float64, col)
	}
	return mat
}

func AddMatrices(m, mTwo Matrix) Matrix {
	// assume number of rows are the same
	mat := New(len(m), len(m[0]))
	for i := range m {
		for j := range m[0] {
			mat[i][j] = m[i][j] + mTwo[i][j]
		}
	}

	return mat
}

func MultScalar(m Matrix, scalar float64) Matrix {
	mat := New(len(m), len(m[0]))
	for i := range m {
		for j := range m[0] {
			mat[i][j] = m[i][j] * scalar
		}
	}
	return mat
}
