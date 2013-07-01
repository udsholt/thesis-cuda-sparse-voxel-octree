#ifndef __RESTLESS_MATH_MAT4X4_H
#define __RESTLESS_MATH_MAT4X4_H

#include <iostream>
#include <iomanip>
#include "Mathlib.h"
#include "VecN.h"
#include "Vec3f.h"

namespace restless
{
	template<class T> 
	class Mat4x4
	{
	protected:
		T _m[16];
	public:

		static const unsigned int ROWS = 4;
		static const unsigned int COLS = 4;
		static const unsigned int SIZE = ROWS * COLS;

		Mat4x4() 
		{
			// Unitialized
		}

		Mat4x4(const T value) 
		{
			fill(value);
		}

		Mat4x4(const VecN<4, T> & v1, const VecN<4, T> & v2, const VecN<4, T> & v3, const VecN<4, T> & v4) {
			Mat4x4<T> & m = * this;
			m(0,0) = v1[0]; m(0,1) = v1[1]; m(0,2) = v1[2]; m(0,3) = v1[3];
			m(1,0) = v2[0]; m(1,1) = v2[1]; m(1,2) = v2[2]; m(1,3) = v2[3];
			m(2,0) = v3[0]; m(2,1) = v3[1]; m(2,2) = v3[2]; m(2,3) = v3[3];
			m(3,0) = v4[0]; m(3,1) = v4[1]; m(3,2) = v4[2]; m(3,3) = v4[3];
		}

		Mat4x4(const Mat4x4<T> & matrix)
		{
			for(int row= 0; row < Mat4x4::ROWS; row++) {
				for(int col = 0; col < Mat4x4::COLS; col++) {
					(*this)(row, col) = matrix(row, col);
				}
			}
		}

		inline
		static Mat4x4<T> Mat4x4::identity()
		{
			Mat4x4<T> matrix = Mat4x4<T>(
				VecN<4,T>( 1,  0,  0,  0),
				VecN<4,T>( 0,  1,  0,  0),
				VecN<4,T>( 0,  0,  1,  0),
				VecN<4,T>( 0,  0,  0,  1)
			);
			return matrix;
		}

		inline
		static Mat4x4<T> Mat4x4::translation(const float tx, const float ty, const float tz)
		{
			Mat4x4<T> matrix = Mat4x4<T>(
				VecN<4,T>( 1,  0,  0, tx),
				VecN<4,T>( 0,  1,  0, ty),
				VecN<4,T>( 0,  0,  1, tz),
				VecN<4,T>( 0,  0,  0,  1)
			);
			return matrix;
		}

		inline
		static Mat4x4<T> Mat4x4::scale(const float sx, const float sy, const float sz)
		{
			Mat4x4<T> matrix = Mat4x4<T>(
				VecN<4,T>( sx,  0,  0, 0),
				VecN<4,T>( 0,  sy,  0, 0),
				VecN<4,T>( 0,  0,  sz, 0),
				VecN<4,T>( 0,  0,  0,  1)
			);
			return matrix;
		}

		inline
		static Mat4x4<T> Mat4x4::ortho(float left, float right, float bottom, float top, float near, float far)
		{
			Mat4x4<T> matrix = Mat4x4<T>(0);

			float r_l = right - left;
			float t_b = top - bottom;
			float f_n = far - near;

			float tx = - (right + left) / (right - left);
			float ty = - (top + bottom) / (top - bottom);
			float tz = - (far + near) / (far - near);

			matrix(0,0) = 2.0f / r_l; matrix(0,1) = 0;          matrix(0,2) = 0;            matrix(0,3) = tx;
			matrix(1,0) = 0;          matrix(1,1) = 2.0f / t_b; matrix(1,2) = 0;            matrix(1,3) = ty;
			matrix(2,0) = 0;          matrix(2,1) = 0;          matrix(2,2) = - 2.0f / f_n; matrix(2,3) = tz;
			matrix(3,0) = 0;          matrix(3,1) = 0;          matrix(3,2) = 0;            matrix(3,3) =  1;

			return matrix;
		}

		// http://stackoverflow.com/questions/2417697/gluperspective-was-removed-in-opengl-3-1-any-replacements
		inline
		static Mat4x4<T> Mat4x4::perspective(float fov, float aspect, float znear, float zfar)
		{
			Mat4x4<T> matrix = Mat4x4<T>(0);

			float xymax = znear * tan(fov * PI_OVER_360);
			float ymin = -xymax;
			float xmin = -xymax;

			float width = xymax - xmin;
			float height = xymax - ymin;

			float depth = zfar - znear;
			float q = -(zfar + znear) / depth;
			float qn = -2 * (zfar * znear) / depth;

			float w = 2 * znear / width;
			w = w / aspect;
			float h = 2 * znear / height;

			matrix(0,0) = w; matrix(0,1) = 0; matrix(0,2) =  0; matrix(0,3) =  0;
			matrix(1,0) = 0; matrix(1,1) = h; matrix(1,2) =  0; matrix(1,3) =  0;
			matrix(2,0) = 0; matrix(2,1) = 0; matrix(2,2) =  q; matrix(2,3) = qn;
			matrix(3,0) = 0; matrix(3,1) = 0; matrix(3,2) = -1; matrix(3,3) =  0;

			return matrix;
		}

		// http://www.felixgers.de/teaching/jogl/perspectiveProjection.html
		inline
		static Mat4x4<T> Mat4x4::frustum(const float left, const float right, const float bottom, const float top, const float near, const float far)
		{
			Mat4x4<T> m = Mat4x4<T>(0);

			m(0,0) = (2.0f * near) / (right - left); 
			m(0,1) = 0.0f; 
			m(0,2) = (right + left) / (right - left);
			m(0,3) = 0;

			m(1,0) = 0.0f; 
			m(1,1) = (2.0f * near) / (top  - bottom); 
			m(1,2) = (top + bottom) / (top - bottom);
			m(1,3) = 0;

			m(2,0) =   0.0f; 
			m(2,1) =   0.0f; 
			m(2,2) = - (far + near) / (far - near);
			m(2,3) = - (2.0f * far * near) / (far - near);

			m(3,0) =  0.0f; 
			m(3,1) =  0.0f; 
			m(3,2) = -1.0f;
			m(3,3) =  0.0f;

			return m;
		}

		// http://www.opengl.org/wiki/GluLookAt_code
		inline
		static Mat4x4<T> Mat4x4::lookat(const Vec3f eye, const Vec3f target, const Vec3f up)
		{
			Mat4x4<T> matrix = Mat4x4<T>(0);

			Vec3f forward = (target - eye).normalize();
			Vec3f side = forward.cross(up).normalize();
			Vec3f upReal = side.cross(forward).normalize();

			matrix(0,0) =      side[0]; matrix(0,1) =      side[1]; matrix(0,2) =      side[2]; matrix(0,3) = 0.0f;
			matrix(1,0) =    upReal[0]; matrix(1,1) =    upReal[1]; matrix(1,2) =    upReal[2]; matrix(1,3) = 0.0f;
			matrix(2,0) = - forward[0]; matrix(2,1) = - forward[1]; matrix(2,2) = - forward[2]; matrix(2,3) = 0.0f;
			matrix(3,0) =         0.0f; matrix(3,1) =         0.0f; matrix(3,2) =         0.0f; matrix(3,3) = 1.0f;

			return matrix * Mat4x4<T>::translation(-eye[0], -eye[1], -eye[2]);
		}

		inline
		static Mat4x4<T> Mat4x4::rotation(restless::Axis axis, T rad)
		{
			Mat4x4<T> matrix = Mat4x4<T>(0);
			
			switch(axis) {
				case AXIS_X:
					matrix(0,0) =  1;
					matrix(1,1) =  cos(rad);
					matrix(1,2) = -sin(rad);
					matrix(2,1) =  sin(rad);
					matrix(2,2) =  cos(rad);
					matrix(3,3) =  1;
					break;
				case AXIS_Y:
					matrix(0,0) =  cos(rad);
					matrix(0,2) =  sin(rad);
					matrix(1,1) =  1;
					matrix(2,0) = -sin(rad);
					matrix(2,2) =  cos(rad);
					matrix(3,3) =  1;
					break;
				case AXIS_Z:
					matrix(0,0) =  cos(rad);
					matrix(0,1) = -sin(rad);
					matrix(1,0) =  sin(rad);
					matrix(1,1) =  cos(rad);
					matrix(2,2) =  1;
					matrix(3,3) =  1;
					break;
			}

			return matrix;
		}

		inline 
		const T * get() const
		{
			return _m;
		}

		// Rotate around a abitrary axis
		// Addapted from Real-Time Rendering second edition, page 43
		static Mat4x4<T> Mat4x4::axisRotation(const VecN<3,T> & r, T theta)
		{
			Mat4x4<T> m = Mat4x4<T>(0);

			T cos_t = cos(theta);
			T sin_t = sin(theta);

			T r_x = r[0];
			T r_y = r[1];
			T r_z = r[2];

			m(0,0) = cos_t + (1 - cos_t) * pow(r_x, 2); 
			m(0,1) = (1 - cos_t) * r_x * r_y - r_z * sin_t; 
			m(0,2) = (1 - cos_t) * r_x * r_z + r_y * sin_t;
			m(0,3) = 0;

			m(1,0) = (1 - cos_t) * r_x * r_y + r_z * sin_t; 
			m(1,1) = cos_t + (1 - cos_t) * pow(r_y, 2); 
			m(1,2) = (1 - cos_t) * r_y * r_z - r_x * sin_t;
			m(1,3) = 0;

			m(2,0) = (1 - cos_t) * r_x * r_z - r_y * sin_t; 
			m(2,1) = (1 - cos_t) * r_y * r_z * r_x * sin_t; 
			m(2,2) = cos_t + (1 - cos_t) * pow(r_z, 2);
			m(2,3) = 0;

			m(3,0) = 0; 
			m(3,1) = 0; 
			m(3,2) = 0;
			m(3,3) = 1;

			return m;
		}

		Mat4x4<T> transpose() const
		{
			Mat4x4<T> m = Mat4x4<T>();

			// Rows becomes columns
			for(int row= 0; row < Mat4x4::ROWS; row++) {
				for(int col = 0; col < Mat4x4::COLS; col++) {
					m(row, col) = (*this)(col, row);
				}
			}
			return m;
		}

		// http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
		// ... originally lifted from MESA implementation of the GLU library.
		Mat4x4<T> inverse() const
		{
			Mat4x4<T> inv = Mat4x4<T>();

			inv[0] = _m[5]  * _m[10] * _m[15] - 
					 _m[5]  * _m[11] * _m[14] - 
					 _m[9]  * _m[6]  * _m[15] + 
					 _m[9]  * _m[7]  * _m[14] +
					 _m[13] * _m[6]  * _m[11] - 
					 _m[13] * _m[7]  * _m[10];

			inv[4] = -_m[4]  * _m[10] * _m[15] + 
					  _m[4]  * _m[11] * _m[14] + 
					  _m[8]  * _m[6]  * _m[15] - 
					  _m[8]  * _m[7]  * _m[14] - 
					  _m[12] * _m[6]  * _m[11] + 
					  _m[12] * _m[7]  * _m[10];

			inv[8] = _m[4]  * _m[9] * _m[15] - 
					 _m[4]  * _m[11] * _m[13] - 
					 _m[8]  * _m[5] * _m[15] + 
					 _m[8]  * _m[7] * _m[13] + 
					 _m[12] * _m[5] * _m[11] - 
					 _m[12] * _m[7] * _m[9];

			inv[12] = -_m[4]  * _m[9] * _m[14] + 
					   _m[4]  * _m[10] * _m[13] +
					   _m[8]  * _m[5] * _m[14] - 
					   _m[8]  * _m[6] * _m[13] - 
					   _m[12] * _m[5] * _m[10] + 
					   _m[12] * _m[6] * _m[9];

			inv[1] = -_m[1]  * _m[10] * _m[15] + 
					  _m[1]  * _m[11] * _m[14] + 
					  _m[9]  * _m[2] * _m[15] - 
					  _m[9]  * _m[3] * _m[14] - 
					  _m[13] * _m[2] * _m[11] + 
					  _m[13] * _m[3] * _m[10];

			inv[5] = _m[0]  * _m[10] * _m[15] - 
					 _m[0]  * _m[11] * _m[14] - 
					 _m[8]  * _m[2] * _m[15] + 
					 _m[8]  * _m[3] * _m[14] + 
					 _m[12] * _m[2] * _m[11] - 
					 _m[12] * _m[3] * _m[10];

			inv[9] = -_m[0]  * _m[9] * _m[15] + 
					  _m[0]  * _m[11] * _m[13] + 
					  _m[8]  * _m[1] * _m[15] - 
					  _m[8]  * _m[3] * _m[13] - 
					  _m[12] * _m[1] * _m[11] + 
					  _m[12] * _m[3] * _m[9];

			inv[13] = _m[0]  * _m[9] * _m[14] - 
					  _m[0]  * _m[10] * _m[13] - 
					  _m[8]  * _m[1] * _m[14] + 
					  _m[8]  * _m[2] * _m[13] + 
					  _m[12] * _m[1] * _m[10] - 
					  _m[12] * _m[2] * _m[9];

			inv[2] = _m[1]  * _m[6] * _m[15] - 
					 _m[1]  * _m[7] * _m[14] - 
					 _m[5]  * _m[2] * _m[15] + 
					 _m[5]  * _m[3] * _m[14] + 
					 _m[13] * _m[2] * _m[7] - 
					 _m[13] * _m[3] * _m[6];

			inv[6] = -_m[0]  * _m[6] * _m[15] + 
					  _m[0]  * _m[7] * _m[14] + 
					  _m[4]  * _m[2] * _m[15] - 
					  _m[4]  * _m[3] * _m[14] - 
					  _m[12] * _m[2] * _m[7] + 
					  _m[12] * _m[3] * _m[6];

			inv[10] = _m[0]  * _m[5] * _m[15] - 
					  _m[0]  * _m[7] * _m[13] - 
					  _m[4]  * _m[1] * _m[15] + 
					  _m[4]  * _m[3] * _m[13] + 
					  _m[12] * _m[1] * _m[7] - 
					  _m[12] * _m[3] * _m[5];

			inv[14] = -_m[0]  * _m[5] * _m[14] + 
					   _m[0]  * _m[6] * _m[13] + 
					   _m[4]  * _m[1] * _m[14] - 
					   _m[4]  * _m[2] * _m[13] - 
					   _m[12] * _m[1] * _m[6] + 
					   _m[12] * _m[2] * _m[5];

			inv[3] = -_m[1] * _m[6] * _m[11] + 
					  _m[1] * _m[7] * _m[10] + 
					  _m[5] * _m[2] * _m[11] - 
					  _m[5] * _m[3] * _m[10] - 
					  _m[9] * _m[2] * _m[7] + 
					  _m[9] * _m[3] * _m[6];

			inv[7] = _m[0] * _m[6] * _m[11] - 
					 _m[0] * _m[7] * _m[10] - 
					 _m[4] * _m[2] * _m[11] + 
					 _m[4] * _m[3] * _m[10] + 
					 _m[8] * _m[2] * _m[7] - 
					 _m[8] * _m[3] * _m[6];

			inv[11] = -_m[0] * _m[5] * _m[11] + 
					   _m[0] * _m[7] * _m[9] + 
					   _m[4] * _m[1] * _m[11] - 
					   _m[4] * _m[3] * _m[9] - 
					   _m[8] * _m[1] * _m[7] + 
					   _m[8] * _m[3] * _m[5];

			inv[15] = _m[0] * _m[5] * _m[10] - 
					  _m[0] * _m[6] * _m[9] - 
					  _m[4] * _m[1] * _m[10] + 
					  _m[4] * _m[2] * _m[9] + 
					  _m[8] * _m[1] * _m[6] - 
					  _m[8] * _m[2] * _m[5];

			T det = _m[0] * inv[0] + _m[1] * inv[4] + _m[2] * inv[8] + _m[3] * inv[12];
			assert(det != 0);
			det = 1.0f / det;

			inv = inv * det;

			//for (i = 0; i < 16; i++) {
			//	invOut[i] = inv[i] * det;
			//}

			return inv;
		}

		/*
		// http://graphics.ics.uci.edu/CS112/web/code/Mat4x4.h
		// Compute the inverse of a 4x4 matrix ASSUMING that its last row is the last
		// row of the 4x4 identity matrix.  This is a common situation, and it makes the
		// computation of the inverse much easier than the general case.  This is called
		// "3x4" as a reminder that the last row is essentially ignored.

		inline Mat4x4 Inverse3x4( const Mat4x4 &M )
		{
			// Compute the determinant of the upper left 3x3 matrix.
			const double det( 
				M(0,0) * ( M(1,1) * M(2,2) - M(1,2) * M(2,1) )
			  - M(0,1) * ( M(1,0) * M(2,2) - M(1,2) * M(2,0) )
			  + M(0,2) * ( M(1,0) * M(2,1) - M(1,1) * M(2,0) )
			  );
    
			// Compute the inverse of the upper left 3x3 matrix.  This is
			// the transpose of the "adjoint" matrix divided by the determinant.
			Mat4x4 W;
			W(0,0) = ( M(1,1) * M(2,2) - M(1,2) * M(2,1) ) / det;
			W(1,0) = ( M(1,2) * M(2,0) - M(1,0) * M(2,2) ) / det;
			W(2,0) = ( M(1,0) * M(2,1) - M(1,1) * M(2,0) ) / det;
 
			W(0,1) = ( M(0,2) * M(2,1) - M(0,1) * M(2,2) ) / det;
			W(1,1) = ( M(0,0) * M(2,2) - M(0,2) * M(2,0) ) / det;
			W(2,1) = ( M(0,1) * M(2,0) - M(0,0) * M(2,1) ) / det;

			W(0,2) = ( M(0,1) * M(1,2) - M(0,2) * M(1,1) ) / det;
			W(1,2) = ( M(0,2) * M(1,0) - M(0,0) * M(1,2) ) / det;
			W(2,2) = ( M(0,0) * M(1,1) - M(0,1) * M(1,0) ) / det;

			W(3,3) = 1;

			// Fill in the translation component of the inverse matrix.
			W.SetTrans( -( W * M.Trans3D() ) );

			return W;
		}
		*/

		void fill(const T value)
		{
			for(int i = 0; i < SIZE; i++) {
				_m[i] = value;
			}
		}

		// Assignment 
		Mat4x4<T> & operator=(const Mat4x4<T> & matrix)
		{
			for(int row= 0; row < ROWS; row++) {
				for(int col = 0; col < COLS; col++) {
					(*this)(row, col) = matrix(row, col);
				}
			}
			return *this;
		}

		// Scalar multiplication
		Mat4x4<T> & operator*=(const T scalar)
		{
			for(int i = 0; i < SIZE; i++) {
				_m[i] *= scalar;
			}
			return *this;
		}

		// Scalar multiplication
		const Mat4x4<T> operator* (const T scalar) const
		{
			// Defined by compund operator
			// http://www.cs.caltech.edu/courses/cs11/material/cpp/donnie/cpp-ops.html
			Mat4x4<T> result = *this;
			result *= scalar;
			return result;
		}
		
		Mat4x4<T> & operator*=(const Mat4x4<T> & otherMatrix)
		{
			(*this) = (*this) * otherMatrix;
			return *this;
		}
	
		// Matrix multiplication
		Mat4x4<T> operator* (const Mat4x4<T> & otherMatrix) const
		{
			// TODO: Very slow implementation, this should be optimized 
			Mat4x4<T> matrix = Mat4x4<T>();
			for(int row= 0; row < ROWS; row++) {
				for(int col = 0; col < COLS; col++) {
					matrix(row, col)  = (*this)(row, 0) * otherMatrix(0, col);
					matrix(row, col) += (*this)(row, 1) * otherMatrix(1, col);
					matrix(row, col) += (*this)(row, 2) * otherMatrix(2, col);
					matrix(row, col) += (*this)(row, 3) * otherMatrix(3, col);
				}
			}
			return matrix;
		}

		const VecN<3, T> operator*(const VecN<3, T> & v) const
		{
			// Bend the rules a bit: mat4x4 * vec3
			// http://www.gamedev.net/community/forums/topic.asp?topic_id=312902
			VecN<3,T> result = VecN<3,T>(0);
			result[0] += (*this)(0, 0) * v[0];
			result[0] += (*this)(0, 1) * v[1];
			result[0] += (*this)(0, 2) * v[2];

			result[1] += (*this)(1, 0) * v[0];
			result[1] += (*this)(1, 1) * v[1];
			result[1] += (*this)(1, 2) * v[2];

			result[2] += (*this)(2, 0) * v[0];
			result[2] += (*this)(2, 1) * v[1];
			result[2] += (*this)(2, 2) * v[2];

			return result;
		}

		const VecN<4, T> operator*(const VecN<4, T> & v) const
		{
			VecN<4,T> result = VecN<4,T>(0);

			result[0] += (*this)(0, 0) * v[0];
			result[0] += (*this)(0, 1) * v[1];
			result[0] += (*this)(0, 2) * v[2];
			result[0] += (*this)(0, 3) * v[3];

			result[1] += (*this)(1, 0) * v[0];
			result[1] += (*this)(1, 1) * v[1];
			result[1] += (*this)(1, 2) * v[2];
			result[1] += (*this)(1, 3) * v[3];

			result[2] += (*this)(2, 0) * v[0];
			result[2] += (*this)(2, 1) * v[1];
			result[2] += (*this)(2, 2) * v[2];
			result[2] += (*this)(2, 3) * v[3];

			result[3] += (*this)(3, 0) * v[0];
			result[3] += (*this)(3, 1) * v[1];
			result[3] += (*this)(3, 2) * v[2];
			result[3] += (*this)(3, 3) * v[3];

			return result;
		}

		// Comparison
		bool operator==(const Mat4x4<T> & matrix) const
		{
			for(int row= 0; row < ROWS; row++) {
				for(int col = 0; col < COLS; col++) {
					if ((*this)(row, col) - matrix(row, col) > 0.0001) {
						return false;
					}
				}
			}
			return true;
		}

		inline
		T operator()(const unsigned row, const unsigned col) const
		{
			assert(row < ROWS && col < COLS);
			return _m[ROWS*col + row];
			//return _m[COLS*row + col]; // original was COLS * row + col
		}

		inline
		T & operator()(const unsigned row, const unsigned col)
		{
			assert(row < ROWS && col < COLS);
			return _m[ROWS*col + row];
			//return _m[COLS*row + col]; // original was COLS * row + col
		}

		inline
		T & operator[](const unsigned index) const
		{
			assert(index < SIZE);
			return _m[index];
		}

		inline
		T & operator[](const unsigned index)
		{
			assert(index < SIZE);
			return _m[index];
		}

	};

	template<class T> 
	std::ostream & operator<<(std::ostream & os, const restless::Mat4x4<T> & matrix)	
	{
		os << std::fixed <<  std::setprecision(5);
		os << "[" << std::endl;
		for(int row= 0; row < restless::Mat4x4<T>::ROWS; row++) {
			for(int col = 0; col < restless::Mat4x4<T>::COLS; col++) {
				os << std::setw(10) << matrix(row, col) << " ";
			}
			os << std::endl;
		}
		os << "]" << std::endl;
		return os;
	}
}

#endif