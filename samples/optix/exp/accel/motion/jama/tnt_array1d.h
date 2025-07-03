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



#ifndef TNT_ARRAY1D_H
#define TNT_ARRAY1D_H

//#include <cstdlib>
#include <iostream>

#ifdef TNT_BOUNDS_CHECK
#include <assert.h>
#endif


namespace TNT
{

template <class T,unsigned int N>
class Array1D 
{

  private:

	  /* ... */
    T v_[N];
    int n_;

    M_DEVICE_HOST void copy_(T* p, const T*  q, int len) const;
    M_DEVICE_HOST void set_(T* begin, T* end, const T& val);
 
 	inline   Array1D(const Array1D &A) = delete;
	inline   Array1D & operator=(const Array1D &A) = delete;
	inline   Array1D & operator=(Array1D &&A) = delete;
	inline   Array1D & ref(Array1D &&A) = delete;
	inline   Array1D(Array1D &&A) = delete;

  public:

    typedef         T   value_type;


	M_DEVICE_HOST          Array1D();
	M_DEVICE_HOST explicit Array1D(int n);
	M_DEVICE_HOST          Array1D(int n, const T &a);
	M_DEVICE_HOST inline   operator T*();
	M_DEVICE_HOST inline   operator const T*();
	M_DEVICE_HOST inline   Array1D & operator=(const T &a);
	M_DEVICE_HOST          Array1D copy() const;
	M_DEVICE_HOST 	     Array1D & inject(const Array1D & A);
	M_DEVICE_HOST inline   T& operator[](int i);
	M_DEVICE_HOST inline   const T& operator[](int i) const;
	M_DEVICE_HOST inline 	 int dim1() const;
	M_DEVICE_HOST inline   int dim() const;
	M_DEVICE_HOST inline 	 void resize(int n);
    M_DEVICE_HOST          ~Array1D();

};


template <class T,unsigned int N>
M_DEVICE_HOST Array1D<T,N>::Array1D()
    : v_()
    , n_( 0 )
{
}

template <class T,unsigned int N>
M_DEVICE_HOST Array1D<T,N>::Array1D(int n) : n_(n)
{
#ifdef TNT_DEBUG
	assert(n < N);
	std::cout << "Created Array1D(int n) \n";
#endif
}

template <class T,unsigned int N>
M_DEVICE_HOST Array1D<T,N>::Array1D(int n, const T &val) : n_(n)
{
#ifdef TNT_DEBUG
	assert(n < N);
	std::cout << "Created Array1D(int n, const T& val) \n";
#endif
	set_(v_, v_+ n, val);

}

template <class T,unsigned int N>
M_DEVICE_HOST inline void Array1D<T,N>::resize(int n)
{
#ifdef TNT_DEBUG	
	assert(n < N);
#endif
	n_ = n;
}

template <class T,unsigned int N>
M_DEVICE_HOST inline Array1D<T,N>::operator T*()
{
	return &(v_[0]);
}


template <class T,unsigned int N>
M_DEVICE_HOST inline Array1D<T,N>::operator const T*()
{
	return &(v_[0]);
}



template <class T,unsigned int N>
M_DEVICE_HOST inline T& Array1D<T,N>::operator[](int i) 
{ 
#ifdef TNT_BOUNDS_CHECK
	assert(i>= 0);
	assert(i < n_);
#endif
	return v_[i]; 
}

template <class T,unsigned int N>
M_DEVICE_HOST inline const T& Array1D<T,N>::operator[](int i) const 
{ 
#ifdef TNT_BOUNDS_CHECK
	assert(i>= 0);
	assert(i < n_);
#endif
	return v_[i]; 
}



template <class T,unsigned int N>
M_DEVICE_HOST Array1D<T,N> & Array1D<T,N>::operator=(const T &a)
{
	set_(v_, v_+n_, a);
	return *this;
}

template <class T,unsigned int N>
M_DEVICE_HOST Array1D<T,N> Array1D<T,N>::copy() const
{
	Array1D A( n_);
	copy_(A.v_, v_, n_);

	return A;
}


template <class T,unsigned int N>
M_DEVICE_HOST Array1D<T,N> & Array1D<T,N>::inject(const Array1D &A)
{
	if (A.n_ == n_)
		copy_(v_, A.v_, n_);

	return *this;
}

template <class T,unsigned int N>
M_DEVICE_HOST inline int Array1D<T,N>::dim1() const { return n_; }

template <class T,unsigned int N>
M_DEVICE_HOST inline int Array1D<T,N>::dim() const { return n_; }

template <class T,unsigned int N>
M_DEVICE_HOST Array1D<T,N>::~Array1D() {}


/* private internal functions */

template <class T,unsigned int N>
M_DEVICE_HOST void Array1D<T,N>::set_(T* begin, T* end, const T& a)
{
	for (T* p=begin; p<end; p++)
		*p = a;

}

template <class T,unsigned int N>
M_DEVICE_HOST void Array1D<T,N>::copy_(T* p, const T* q, int len) const
{
	T *end = p + len;
	while (p<end )
		*p++ = *q++;

}


} /* namespace TNT */

#endif
/* TNT_ARRAY1D_H */

