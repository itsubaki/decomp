package matrix

import (
	"iter"
	"math/cmplx"

	"github.com/itsubaki/decomp/epsilon"
)

// Matrix is a matrix of complex128.
type Matrix struct {
	Rows int
	Cols int
	Data []complex128
}

// New returns a new matrix of complex128.
func New(z ...[]complex128) *Matrix {
	rows := len(z)
	var cols int
	if rows > 0 {
		cols = len(z[0])
	}

	data := make([]complex128, rows*cols)
	for i := range rows {
		copy(data[i*cols:(i+1)*cols], z[i])
	}

	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// Zero returns a zero matrix.
func Zero(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]complex128, rows*cols),
	}
}

// ZeroLike returns a zero matrix of same size as m.
func ZeroLike(m *Matrix) *Matrix {
	rows, cols := m.Dimension()
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]complex128, rows*cols),
	}
}

// Identity returns an identity matrix.
func Identity(size int) *Matrix {
	m := Zero(size, size)
	for i := range size {
		m.Set(i, i, 1)
	}

	return m
}

// Clone returns a clone of matrix.
func (m *Matrix) Clone() *Matrix {
	out := ZeroLike(m)
	copy(out.Data, m.Data)
	return out
}

// Set sets a value of matrix at (i,j).
func (m *Matrix) Set(i, j int, z complex128) {
	m.Data[i*m.Cols+j] = z
}

// At returns a value of matrix at (i,j).
func (m *Matrix) At(i, j int) complex128 {
	return m.Data[i*m.Cols+j]
}

// AddAt adds a value of matrix at (i,j).
func (m *Matrix) AddAt(i, j int, z complex128) {
	m.Data[i*m.Cols+j] += z
}

// SubAt subtracts a value of matrix at (i,j).
func (m *Matrix) SubAt(i, j int, z complex128) {
	m.Data[i*m.Cols+j] -= z
}

// MulAt multiplies a value of matrix at (i,j).
func (m *Matrix) MulAt(i, j int, z complex128) {
	m.Data[i*m.Cols+j] *= z
}

// DivAt divides a value of matrix at (i,j).
func (m *Matrix) DivAt(i, j int, z complex128) {
	m.Data[i*m.Cols+j] /= z
}

// Row returns a row of matrix at (i).
func (m *Matrix) Row(i int) []complex128 {
	return row(m.Data, m.Cols, i)
}

// Dimension returns a dimension of matrix.
func (m *Matrix) Dimension() (rows int, cols int) {
	return m.Rows, m.Cols
}

// MatMul returns the matrix product of m and n.
// A.MatMul(B) is AB.
func (m *Matrix) MatMul(n *Matrix) *Matrix {
	a, b := m.Dimension()
	_, p := n.Dimension()

	out := Zero(a, p)
	for i := range a {
		for k := range b {
			mik := m.Data[i*b+k]
			for j := range p {
				out.Data[i*p+j] += mik * n.Data[k*p+j]
			}
		}
	}

	return out
}

// Dagger returns conjugate transpose matrix.
func (m *Matrix) Dagger() *Matrix {
	out := ZeroLike(m)
	for i := range m.Rows {
		for j := range m.Cols {
			out.Set(j, i, cmplx.Conj(m.At(i, j)))
		}
	}

	return out
}

// Inverse returns an inverse matrix of m.
func (m *Matrix) Inverse(eps ...float64) *Matrix {
	p, q := m.Dimension()
	mm := m.Clone()
	e := epsilon.E13(eps...)

	out := Identity(p)
	for i := range p {
		if cmplx.Abs(mm.At(i, i)) < e {
			// swap rows
			for r := i + 1; r < p; r++ {
				if cmplx.Abs(mm.At(r, i)) < e {
					continue
				}

				mm = mm.Swap(i, r)
				out = out.Swap(i, r)
				break
			}
		}

		c := 1 / mm.At(i, i)
		for j := range q {
			mm.MulAt(i, j, c)
			out.MulAt(i, j, c)
		}

		for j := range q {
			if i == j {
				continue
			}

			c := mm.At(j, i)
			for k := range q {
				mm.AddAt(j, k, -c*mm.At(i, k))
				out.AddAt(j, k, -c*out.At(i, k))
			}
		}
	}

	return out
}

// Swap returns a matrix of m with i-th and j-th rows swapped.
func (m *Matrix) Swap(i, j int) *Matrix {
	data := make([]complex128, len(m.Data))
	copy(data, m.Data)

	i0, i1 := i*m.Cols, (i+1)*m.Cols
	j0, j1 := j*m.Cols, (j+1)*m.Cols

	tmp := make([]complex128, m.Cols)
	copy(tmp, data[i0:i1])
	copy(data[i0:i1], data[j0:j1])
	copy(data[j0:j1], tmp)

	return &Matrix{
		Rows: m.Rows,
		Cols: m.Cols,
		Data: data,
	}
}

// Equals returns true if m equals n.
// If eps is not given, epsilon.E13 is used.
func (m *Matrix) Equals(n *Matrix, eps ...float64) bool {
	p, q := m.Dimension()
	a, b := n.Dimension()

	if a != p || b != q {
		return false
	}

	e := epsilon.E13(eps...)
	for i := range m.Data {
		if cmplx.Abs(m.Data[i]-n.Data[i]) > e {
			return false
		}
	}

	return true
}

// IsSquare returns true if m is square matrix.
func (m *Matrix) IsSquare() bool {
	return m.Rows == m.Cols
}

// IsUnitary returns true if m is unitary matrix.
func (m *Matrix) IsUnitary(eps ...float64) bool {
	return m.IsSquare() && m.MatMul(m.Dagger()).Equals(Identity(m.Rows), eps...)
}

// IsUpperTriangular returns true if m is upper triangular matrix.
func (m *Matrix) IsUpperTriangular(eps ...float64) bool {
	if !m.IsSquare() {
		return false
	}

	e := epsilon.E13(eps...)
	for i := 1; i < m.Rows; i++ {
		for j := range i {
			if cmplx.Abs(m.At(i, j)) > e {
				return false
			}
		}
	}

	return true
}

// IsHessenberg returns true if m is Hessenberg matrix.
func (m *Matrix) IsHessenberg(eps ...float64) bool {
	if !m.IsSquare() {
		return false
	}

	e := epsilon.E13(eps...)
	for i := 2; i < m.Rows; i++ {
		for j := range i - 1 {
			if cmplx.Abs(m.At(i, j)) > e {
				return false
			}
		}
	}

	return true
}

// IsDiagonal returns true if m is diagonal matrix.
func (m *Matrix) IsDiagonal(eps ...float64) bool {
	if !m.IsSquare() {
		return false
	}

	e := epsilon.E13(eps...)
	for i := range m.Rows {
		for j := range m.Cols {
			if i == j {
				continue
			}

			if cmplx.Abs(m.At(i, j)) > e {
				return false
			}
		}
	}

	return true
}

// Seq2 returns a sequence of rows.
func (m *Matrix) Seq2() iter.Seq2[int, []complex128] {
	return func(yield func(int, []complex128) bool) {
		for i := range m.Rows {
			if !yield(i, m.Row(i)) {
				return
			}
		}
	}
}

// MatMul returns a matrix product of m1, m2, ..., mn.
// MatMul(A, B, C, D, ...) is ABCD....
func MatMul(m ...*Matrix) *Matrix {
	out := m[0]
	for i := 1; i < len(m); i++ {
		out = out.MatMul(m[i])
	}

	return out
}

func row[T any](arr []T, cols, i int) []T {
	return arr[i*cols : (i+1)*cols]
}
