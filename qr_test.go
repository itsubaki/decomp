package decomp_test

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/itsubaki/decomp"
	"github.com/itsubaki/decomp/epsilon"
	"github.com/itsubaki/decomp/matrix"
)

func rx(theta float64) *matrix.Matrix {
	v := complex(theta/2, 0)
	return matrix.New(
		[]complex128{cmplx.Cos(v), -1i * cmplx.Sin(v)},
		[]complex128{-1i * cmplx.Sin(v), cmplx.Cos(v)},
	)
}

func ry(theta float64) *matrix.Matrix {
	v := complex(theta/2, 0)
	return matrix.New(
		[]complex128{cmplx.Cos(v), -cmplx.Sin(v)},
		[]complex128{cmplx.Sin(v), cmplx.Cos(v)},
	)
}

func rz(theta float64) *matrix.Matrix {
	v := complex(0, theta/2)
	return matrix.New(
		[]complex128{cmplx.Exp(-1 * v), 0},
		[]complex128{0, cmplx.Exp(v)},
	)
}

func ExampleQR_rank1() {
	isZero := func(row []complex128) bool {
		for _, v := range row {
			if cmplx.Abs(v) < epsilon.E13() {
				continue
			}

			return false
		}

		return true
	}

	a := matrix.New(
		[]complex128{1, 2, 3},
		[]complex128{2, 4, 6},
		[]complex128{3, 6, 9},
	)

	for _, qr := range []decomp.QRFunc{
		decomp.QR,
		decomp.QRHH,
	} {
		_, r := qr(a)
		for i, row := range r.Seq2() {
			fmt.Println(i, ":", isZero(row))
		}
	}

	// Output:
	// 0 : false
	// 1 : true
	// 2 : true
	// 0 : false
	// 1 : true
	// 2 : true
}

func TestQR(t *testing.T) {
	cases := []struct {
		in *matrix.Matrix
	}{
		{
			matrix.New(
				[]complex128{1, 0},
				[]complex128{0, 1},
			),
		},
		{
			matrix.New(
				[]complex128{0, 1},
				[]complex128{1, 0},
			),
		},
		{
			matrix.New(
				[]complex128{0, -1i},
				[]complex128{1i, 0},
			),
		},
		{
			matrix.New(
				[]complex128{1, 0},
				[]complex128{0, -1},
			),
		},
		{
			matrix.New(
				[]complex128{1 / math.Sqrt2, 1 / math.Sqrt2},
				[]complex128{1 / math.Sqrt2, -1 / math.Sqrt2},
			),
		},
		{
			matrix.New(
				[]complex128{1, 0},
				[]complex128{0, 1i},
			),
		},
		{
			matrix.New(
				[]complex128{1, 0},
				[]complex128{0, cmplx.Exp(1i * math.Pi / 4)},
			),
		},
		{
			matrix.New(
				[]complex128{1, 0},
				[]complex128{0, cmplx.Exp(complex(0, rand.Float64()))},
			),
		},
		{
			rx(rand.Float64()),
		},
		{
			ry(rand.Float64()),
		},
		{
			rz(rand.Float64()),
		},
		{
			matrix.New(
				[]complex128{0, 0, 0, 1},
				[]complex128{0, 0, 1, 0},
				[]complex128{0, 1, 0, 0},
				[]complex128{1, 0, 0, 0},
			),
		},
		{
			matrix.New(
				[]complex128{1, 0, 0, 0},
				[]complex128{0, 1, 0, 0},
				[]complex128{0, 0, 0, 1},
				[]complex128{0, 0, 1, 0},
			),
		},
		{
			matrix.New(
				[]complex128{1, 0, 0, 0},
				[]complex128{0, 1, 0, 0},
				[]complex128{0, 0, 1, 0},
				[]complex128{0, 0, 0, 1},
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
	}

	for _, qr := range []decomp.QRFunc{
		decomp.QR,
		decomp.QRHH,
	} {
		for _, c := range cases {
			Q, R := qr(c.in)

			if !Q.IsUnitary() {
				t.Errorf("Q is not unitary")
			}

			if !R.IsUpperTriangular() {
				t.Errorf("R is not upper triangular")
			}

			if !matrix.MatMul(Q, R).Equals(c.in) {
				t.Errorf("matmul(Q, R) does not equal a")
			}
		}
	}
}
