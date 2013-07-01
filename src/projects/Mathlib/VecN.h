#ifndef __RESTLESS_MATH_VECN_H
#define __RESTLESS_MATH_VECN_H

#include <iostream>
#include <iomanip>
#include <assert.h>

namespace restless
{
	template<unsigned int N, class T> 
	class VecN
	{
	protected:
		T _data[N];

	public:
		VecN<N,T>() {}
		VecN<N,T>(const T value) {
			fill(value);
		}

		VecN<N,T>(const VecN<N,T> & v)
		{
			*this = v;
		}

		VecN<N,T>(const T x, const T y, const T z, const T w)
		{
			assert(N == 4);
			_data[0] = x;
			_data[1] = y;
			_data[2] = z;
			_data[3] = w;
		}

		inline 
		const T * get() const
		{
			return _data;
		}

		void fill(const T value)
		{
			for(int i = 0; i < N; i++) {
				_data[i] = value;
			}
		}

		inline
		T dot(const VecN<N,T> & v) const
		{
			T result = 0;
			for(int i = 0; i < N; i++) {
				result += _data[i] * v[i];
			}
			return result;
		}

		inline
		double magnitudeSquared() const
		{
			return (double)(*this).dot(*this);
		}

		inline 
		double magnitude() const 
		{
			return sqrt((double)magnitudeSquared());
		}

		VecN<N,T> normalize() const
		{
			VecN<N,T> result = *this;
			
			double magnitude = result.magnitude();
			
			for(int i = 0; i < N; i++) {
				result[i] = (T)(result[i] / magnitude); // do the calculation cast cast to whatever is in the vector
			}			

			return result;
		}

		// Vector addition
		VecN<N,T> & operator+=(const VecN<N,T> & v)
		{
			for(int i = 0; i < N; i++) {
				_data[i] += v[i];
			}
			return *this;
		}

		const VecN<N,T> operator+ (const VecN<N,T> & v) const
		{
			// Defined by compund operator
			// http://www.cs.caltech.edu/courses/cs11/material/cpp/donnie/cpp-ops.html
			VecN<N,T> result = *this;
			result += v;
			return result;
		}

		// Vector subtraction
		VecN<N,T> & operator-=(const VecN<N,T> & v)
		{
			for(int i = 0; i < N; i++) {
				_data[i] -= v[i];
			}
			return *this;
		}

		const VecN<N,T> operator- (const VecN<N,T> & v) const
		{
			// Defined by compund operator
			// http://www.cs.caltech.edu/courses/cs11/material/cpp/donnie/cpp-ops.html
			VecN<N,T> result = *this;
			result -= v;
			return result;
		}

		// Negate operator
		const VecN<N,T> operator- () const
		{
			VecN<N,T> result = *this;
			for(int i = 0; i < N; i++) {
				result[i] = -result[i];
			}
			return result;
		}

		// Vector multiplication
		VecN<N,T> & operator*=(const VecN<N,T> & v)
		{
			for(int i = 0; i < N; i++) {
				_data[i] *= v[i];
			}
			return *this;
		}

		const VecN<N,T> operator* (const VecN<N,T> & v) const
		{
			// Defined by compund operator
			// http://www.cs.caltech.edu/courses/cs11/material/cpp/donnie/cpp-ops.html
			VecN<N,T> result = *this;
			result *= v;
			return result;
		}

		// Vector division
		VecN<N,T> & operator/=(const VecN<N,T> & v)
		{
			for(int i = 0; i < N; i++) {
				if (v[i] == 0) {
					_data[i] = 0;
				} else {
					_data[i] /= v[i];
				}
			}
			return *this;
		}

		const VecN<N,T> operator/ (const VecN<N,T> & v) const
		{
			// Defined by compund operator
			// http://www.cs.caltech.edu/courses/cs11/material/cpp/donnie/cpp-ops.html
			VecN<N,T> result = *this;
			result /= v;
			return result;
		}

		// Scalar division
		VecN<N,T> & operator/=(const T scalar)
		{
			// Calculate a multiplier rather then dividing several times
			T multiplier = 0;
			if (scalar != 0) {
				multiplier = (T)1.0 / scalar; // Casting to T could give some strange results .. :)
			}
			(*this) *= multiplier;
			return *this;
		}

		const VecN<N,T> operator/ (const T scalar) const
		{
			// Defined by compund operator
			// http://www.cs.caltech.edu/courses/cs11/material/cpp/donnie/cpp-ops.html
			VecN<N,T> result = *this;
			result /= scalar;
			return result;
		}


		// Scalar multiplication
		VecN<N,T> & operator*=(const T scalar)
		{
			for(int i = 0; i < N; i++) {
				_data[i] *= scalar;
			}
			return *this;
		}

		const VecN<N,T> operator* (const T scalar) const
		{
			// Defined by compund operator
			// http://www.cs.caltech.edu/courses/cs11/material/cpp/donnie/cpp-ops.html
			VecN<N,T> result = *this;
			result *= scalar;
			return result;
		}

		// Assignment 
		VecN<N,T> & operator=(const VecN<N,T> & vector)
		{
			for(int i = 0; i < N; i++) {
				_data[i] = vector[i];
			}
			return *this;
		}

		// Comparison
		bool operator==(VecN<N,T> & vector) const
		{	
			for(int i = 0; i < N; i++) {
				if ((*this)[i] != vector[i]) {
					return false;
				}
			}
			return true;
		}

		inline
		T operator[](const unsigned index) const
		{
			assert(index < N);
			return _data[index];
		}

		inline
		T & operator[](const unsigned index)
		{
			assert(index < N);
			return _data[index];
		}

	};



	template<unsigned int N, class T> 
	std::ostream & operator<<(std::ostream & os, const restless::VecN<N,T> & v)	
	{
		os << std::setprecision(5) << std::fixed;
		os << "[";
		for(int i = 0; i < N; i++) {
			os << std::setw(10) << v[i] << " ";
		}
		os << "]";
		return os;
	}
}

#endif