package decomp_test

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/itsubaki/decomp"
	"github.com/itsubaki/decomp/matrix"
)

func ExampleEigenJacobi_pow0p5() {
	x := matrix.New(
		[]complex128{0, 1},
		[]complex128{1, 0},
	)

	V, D := decomp.EigenJacobi(x, 10)
	D.Fdiag(func(v complex128) complex128 { return cmplx.Pow(v, 0.5) })

	sqrtx := matrix.MatMul(V, D, V.Dagger())
	for _, row := range sqrtx.Seq2() {
		fmt.Printf("%.3f\n", row)
	}

	sqrtx2 := matrix.MatMul(sqrtx, sqrtx)
	fmt.Println(sqrtx2.Equal(x))

	// Output:
	// [(0.500+0.500i) (0.500-0.500i)]
	// [(0.500-0.500i) (0.500+0.500i)]
	// true
}

func ExampleEigenJacobi_pow1p5() {
	x := matrix.New(
		[]complex128{0, 1},
		[]complex128{1, 0},
	)

	V, D := decomp.EigenJacobi(x, 10)
	D.Fdiag(func(v complex128) complex128 { return cmplx.Pow(v, 1.5) })

	x1p5 := matrix.MatMul(V, D, V.Dagger())
	for _, row := range x1p5.Seq2() {
		fmt.Printf("%.3f\n", row)
	}

	x1p52 := matrix.MatMul(x1p5, x1p5)
	fmt.Println(x1p52.Equal(x))

	// Output:
	// [(0.500-0.500i) (0.500+0.500i)]
	// [(0.500+0.500i) (0.500-0.500i)]
	// true
}

func ExampleEigenJacobi_exp() {
	exp := func(x *matrix.Matrix, theta float64) *matrix.Matrix {
		V, D := decomp.EigenJacobi(x, 10)
		D.Fdiag(func(v complex128) complex128 { return cmplx.Exp(-1i * complex(theta/2, 0) * v) })
		return matrix.MatMul(V, D, V.Dagger())
	}

	x := matrix.New(
		[]complex128{0, 1},
		[]complex128{1, 0},
	)

	theta := rand.Float64()
	expiX := exp(x, theta)
	fmt.Println(expiX.Equal(rx(theta)))

	// Output:
	// true
}

func TestEigenQR(t *testing.T) {
	cases := []struct {
		in *matrix.Matrix
	}{
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
			P, D := decomp.EigenQR(c.in, qr, 20)

			if !D.IsDiagonal() {
				t.Errorf("D is not diagonal")
			}

			if !matrix.MatMul(P, D, P.Inverse()).Equal(c.in) {
				t.Errorf("P * D * P^-1 does not equal a")
			}
		}
	}
}

func TestEigenUpperT(t *testing.T) {
	cases := []struct {
		in *matrix.Matrix
	}{
		{
			matrix.New(
				[]complex128{1, 2},
				[]complex128{0, 3},
			),
		},
		{
			matrix.New(
				[]complex128{1, 0, 0},
				[]complex128{0, 2, 0},
				[]complex128{0, 0, 3},
			),
		},
		{
			matrix.New(
				[]complex128{1, 2, 3, 4, 5},
				[]complex128{0, 2, 3, 4, 5},
				[]complex128{0, 0, 3, 4, 5},
				[]complex128{0, 0, 0, 4, 5},
				[]complex128{0, 0, 0, 0, 5},
			),
		},
		{
			matrix.New(
				[]complex128{1 + 1i, 2 - 1i, 3 + 0.5i},
				[]complex128{0, 2 + 2i, 1 - 0.5i},
				[]complex128{0, 0, 3 - 1i},
			),
		},
		{

			matrix.New(
				[]complex128{5, 0, 0, 1},
				[]complex128{0, 3, 0, 0},
				[]complex128{0, 0, 2, 0},
				[]complex128{0, 0, 0, 1},
			),
		},
		{
			matrix.New(
				[]complex128{10, 0, 0, 0, 0, 2},
				[]complex128{0, 9, 0, 0, 0, 0},
				[]complex128{0, 0, 8, 0, 0, 0},
				[]complex128{0, 0, 0, 7, 0, 0},
				[]complex128{0, 0, 0, 0, 6, 0},
				[]complex128{0, 0, 0, 0, 0, 5},
			),
		},
	}

	for _, c := range cases {
		P, D := decomp.EigenUpperTriangular(c.in)

		if !D.IsDiagonal() {
			t.Errorf("D is not diagonal")
		}

		if !matrix.MatMul(P, D, P.Inverse()).Equal(c.in) {
			t.Errorf("P * D * P^-1 does not equal t")
		}
	}
}

func TestEigenJacobi(t *testing.T) {
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
	}

	for _, c := range cases {
		if !c.in.Equal(c.in.Dagger()) {
			t.Errorf("input is not Hermitian")
		}

		V, D := decomp.EigenJacobi(c.in, 10)

		if !V.IsUnitary() {
			t.Errorf("V * V^dagger does not equal I")
		}

		if !D.IsDiagonal() {
			t.Errorf("D is not diagonal")
		}

		if !matrix.MatMul(V, D, V.Dagger()).Equal(c.in) {
			t.Errorf("V * D * V^dagger does not equal a")
			for _, row := range V.Seq2() {
				t.Log(row)
			}
		}
	}
}
