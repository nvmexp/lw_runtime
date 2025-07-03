/*
*
* Template Numerical Toolkit (TNT)
*
* Mathematical and Computational Sciences Division
* National Institute of Technology,
* Gaithersburg, MD USA
*
*
* This software was developed at the National Institute of Standards and
* Technology (NIST) by employees of the Federal Government in the course
* of their official duties. Pursuant to title 17 Section 105 of the
* United States Code, this software is not subject to copyright protection
* and is in the public domain. NIST assumes no responsibility whatsoever for
* its use by other parties, and makes no guarantees, expressed or implied,
* about its quality, reliability, or any other characteristic.
*
*/



#ifndef TNT_ARRAY2D_H
#define TNT_ARRAY2D_H

#include <cstdlib>
#include <iostream>
#ifdef TNT_BOUNDS_CHECK
#include <assert.h>
#endif

#include "tnt_array1d.h"

namespace TNT
{

template <class T,unsigned int M,unsigned int N>
class Array2D 
{


  private:

  	Array1D<T,N*M> data_;
	int m_;
    int n_;

	inline Array2D(const Array2D &A) = delete;
	inline Array2D & operator=(Array2D &A) = delete;
	inline Array2D & operator=(Array2D &&A) = delete;
	inline Array2D(Array2D &&A) = delete;

  public:

    typedef         T   value_type;
	M_DEVICE_HOST        Array2D();
	M_DEVICE_HOST        Array2D(int m, int n);
	M_DEVICE_HOST        Array2D(int m, int n, const T &a);
	M_DEVICE_HOST inline operator T**();
	M_DEVICE_HOST inline operator const T**();
	M_DEVICE_HOST inline Array2D & operator=(const T &a);
	M_DEVICE_HOST        Array2D copy() const;
	M_DEVICE_HOST 	   Array2D & inject(const Array2D & A);
	M_DEVICE_HOST inline T* operator[](int i);
	M_DEVICE_HOST inline const T* operator[](int i) const;
	M_DEVICE_HOST inline int dim1() const;
	M_DEVICE_HOST inline int dim2() const;
    M_DEVICE_HOST  ~Array2D();

    M_DEVICE_HOST inline void resize(int m, int n);

};


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST Array2D<T,M,N>::Array2D() : data_(), m_(0), n_(0) {} 

template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST Array2D<T,M,N>::Array2D(int m, int n) : data_(m*n), m_(m), n_(n)
{
    resize( m, n );
}


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST Array2D<T,M,N>::Array2D(int m, int n, const T &val) : data_(m*n), m_(m), n_(n) 
{
#ifdef TNT_BOUNDS_CHECK	
	assert( m < M );
	assert( n < N );
#endif
    data_ = val;
}


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST inline void Array2D<T,M,N>::resize(int m, int n)
{
#ifdef TNT_BOUNDS_CHECK	
	assert( m < M );
	assert( n < N );
#endif
	data_.resize(m*n);
	m_ = m;
	n_ = n;
}

template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST inline T* Array2D<T,M,N>::operator[](int i) 
{ 
#ifdef TNT_BOUNDS_CHECK
	assert(i >= 0);
	assert(i < m_);
#endif

	return &data_[i*n_]; 
}


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST inline const T* Array2D<T,M,N>::operator[](int i) const
{ 
#ifdef TNT_BOUNDS_CHECK
	assert(i >= 0);
	assert(i < m_);
#endif

	return &data_[i*n_];
}

template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST Array2D<T,M,N> & Array2D<T,M,N>::operator=(const T &a)
{
	/* non-optimzied, but will work with subarrays in future verions */

	for (int i=0; i<m_; i++)
		for (int j=0; j<n_; j++)
		operator[](i)[j] = a;
	return *this;
}


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST Array2D<T,M,N> Array2D<T,M,N>::copy() const
{
	Array2D A(m_, n_);

	for (int i=0; i<m_; i++)
		for (int j=0; j<n_; j++)
			A[i][j] = operator[](i)[j];


	return A;
}


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST Array2D<T,M,N> & Array2D<T,M,N>::inject(const Array2D &A)
{
	if (A.m_ == m_ &&  A.n_ == n_)
	{
		for (int i=0; i<m_; i++)
			for (int j=0; j<n_; j++)
				operator[](i)[j] = A[i][j];
	}
	return *this;
}


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST inline int Array2D<T,M,N>::dim1() const { return m_; }

template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST inline int Array2D<T,M,N>::dim2() const { return n_; }


template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST Array2D<T,M,N>::~Array2D() {}




template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST inline Array2D<T,M,N>::operator T**()
{
	return &(data_[0]);
}
template <class T,unsigned int M,unsigned int N>
M_DEVICE_HOST inline Array2D<T,M,N>::operator const T**()
{
	return &(data_[0]);
}


} /* namespace TNT */

#endif
/* TNT_ARRAY2D_H */

