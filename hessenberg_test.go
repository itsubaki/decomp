package decomp_test

import (
	"testing"

	"github.com/itsubaki/decomp"
	"github.com/itsubaki/decomp/matrix"
)

func TestHessenberg(t *testing.T) {
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
				[]complex128{1, 2, 3},
				[]complex128{2, 5, 6},
				[]complex128{3, 6, 9},
			),
		},
		{
			matrix.New(
				[]complex128{2 + 0i, 1 - 1i, 0},
				[]complex128{1 + 1i, 3 + 0i, 4 - 2i},
				[]complex128{0, 4 + 2i, 1 + 0i},
			),
		},
		{
			matrix.Identity(3),
		},
		{
			matrix.New(
				[]complex128{1 + 1i, 2, 3 - 1i, 4},
				[]complex128{0, 5 + 2i, 6, 7 - 1i},
				[]complex128{0, 0, 8 + 1i, 9},
				[]complex128{0, 0, 0, 10 + 3i},
			),
		},
	}

	for _, c := range cases {
		Q, T := decomp.Hessenberg(c.in)

		if !Q.IsUnitary() {
			t.Errorf("Q is not unitary")
		}

		if !T.IsHessenberg() {
			t.Errorf("T is not in Hessenberg form")
		}

		if !matrix.MatMul(Q, T, Q.Dagger()).Equal(c.in) {
			t.Errorf("Q * T * Q^dagger does not equal a")
		}
	}
}
