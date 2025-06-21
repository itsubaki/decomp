package decomp_test

import (
	"fmt"
	"math/cmplx"

	"github.com/itsubaki/decomp"
	"github.com/itsubaki/decomp/matrix"
)

func ExampleParlett() {
	pow := func(p complex128) (decomp.ParlettF, decomp.ParlettF) {
		return func(z complex128) complex128 {
				return cmplx.Pow(z, p)
			}, func(z complex128) complex128 {
				return p * cmplx.Pow(z, p-1)
			}
	}

	t := matrix.New(
		[]complex128{2, 2, 3},
		[]complex128{0, 2, 5},
		[]complex128{0, 0, 3},
	)

	f, df := pow(2)
	t2 := decomp.Parlett(t, f, df)
	fmt.Println(t2.Equals(matrix.MatMul(t, t)))

	// Output:
	// true
}
