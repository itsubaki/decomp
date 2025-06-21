package decomp

import "github.com/itsubaki/decomp/matrix"

// QRFunc is a function type that performs QR decomposition on a given matrix.
// It returns Q (orthonormal columns) and R (upper triangular) such that A = Q * R.
type QRFunc func(a *matrix.Matrix, eps ...float64) (q *matrix.Matrix, r *matrix.Matrix)

var (
	_ QRFunc = QR
	_ QRFunc = QRHH
)

// Schur performs the Schur decomposition of matrix a using iterative QR decomposition.
// It returns Q (unitary) and T (upper triangular) such that A = Q * T * Q^dagger.
func Schur(a *matrix.Matrix, qr QRFunc, iter int, eps ...float64) (q *matrix.Matrix, t *matrix.Matrix) {
	q, t = matrix.Identity(a.Rows), a.Clone()

	for range iter {
		qk, rk := qr(t, eps...)
		t = rk.MatMul(qk)
		q = q.MatMul(qk)

		if t.IsUpperTriangular(eps...) {
			break
		}
	}

	return q, t
}
