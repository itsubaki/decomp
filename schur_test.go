package decomp_test

import (
	"testing"

	"github.com/itsubaki/decomp"
	"github.com/itsubaki/decomp/matrix"
)

func TestSchur(t *testing.T) {
	cases := []struct {
		in *matrix.Matrix
	}{
		{
			matrix.New(
				[]complex128{1 + 1i, 2 - 1i},
				[]complex128{3 + 4i, 4},
			),
		},
		{
			matrix.New(
				[]complex128{0 + 1i, 1 - 1i, 2},
				[]complex128{0, 3 + 3i, 1 + 1i},
				[]complex128{0, 0, 4 - 4i},
			),
		},
		{
			matrix.New(
				[]complex128{1, 2},
				[]complex128{3, 4},
			),
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{3, 4, 5},
				[]complex128{7, 8, 10},
			),
		},
		{
			matrix.New(
				[]complex128{1, 2},
				[]complex128{2, 3},
			),
		},
		{
			matrix.New(
				[]complex128{1, 1i},
				[]complex128{-1i, 3},
			),
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{0, 4, 5},
				[]complex128{0, 0, 6},
			),
		},
	}

	for _, qr := range []decomp.QRFunc{
		decomp.QR,
		decomp.QRHH,
	} {
		for _, c := range cases {
			Q, T := decomp.Schur(c.in, qr, 20)

			if !Q.IsUnitary() {
				t.Errorf("Q is not unitary")
			}

			if !T.IsUpperTriangular() {
				t.Errorf("T is not upper triangular")
			}

			if !matrix.MatMul(Q, T, Q.Dagger()).Equals(c.in) {
				t.Errorf("Q * T * Q^dagger does not equal a")
			}
		}
	}
}
