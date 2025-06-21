package matrix_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/decomp/matrix"
)

func ExampleMatrix_IsDiagonal() {
	x := matrix.New(
		[]complex128{1, 0},
		[]complex128{0, 2},
		[]complex128{3, 4},
	)

	fmt.Println(x.IsDiagonal())

	// Output:
	// false
}

func TestIsUpperTriangular(t *testing.T) {
	cases := []struct {
		in   *matrix.Matrix
		want bool
	}{
		{
			matrix.New(
				[]complex128{1, 2},
				[]complex128{0, 3},
			),
			true,
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{0, 4, 5},
				[]complex128{0, 0, 6},
			),
			true,
		},
		{
			matrix.New(
				[]complex128{1, 2},
				[]complex128{3, 4},
			),
			false,
		},
		{
			matrix.New(
				[]complex128{1, 2},
				[]complex128{0, 4},
				[]complex128{0, 6},
			),
			false,
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{0, 4, 5},
			),
			false,
		},
		{
			matrix.New(
				[]complex128{complex(1, 1), complex(2, 2)},
				[]complex128{0, complex(3, -1)},
			),
			true,
		},
		{
			matrix.New(
				[]complex128{complex(1, 0), complex(2, 0)},
				[]complex128{complex(1, 1), complex(3, 0)},
			),
			false,
		},
	}

	for _, c := range cases {
		if c.in.IsUpperTriangular() != c.want {
			t.Fail()
		}
	}
}

func TestIsHessenberg(t *testing.T) {
	cases := []struct {
		in   *matrix.Matrix
		want bool
	}{
		{
			matrix.New(
				[]complex128{1, 2},
				[]complex128{3, 4},
			),
			true,
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{4, 5, 6},
				[]complex128{0, 7, 8},
			),
			true,
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{4, 5, 6},
				[]complex128{9, 7, 8},
			),
			false,
		},
		{
			matrix.Identity(4),
			true,
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{0, 4, 5},
				[]complex128{0, 0, 6},
			),
			true,
		},
		{
			matrix.New(
				[]complex128{1, 0, 0},
				[]complex128{2, 3, 0},
				[]complex128{4, 5, 6},
			),
			false,
		},
		{
			matrix.New(
				[]complex128{1, 2, 3},
				[]complex128{4, 5, 6},
				[]complex128{0, 7, 8},
				[]complex128{0, 0, 9},
			),
			false,
		},
	}

	for _, c := range cases {
		if c.in.IsHessenberg() != c.want {
			t.Fail()
		}
	}
}
