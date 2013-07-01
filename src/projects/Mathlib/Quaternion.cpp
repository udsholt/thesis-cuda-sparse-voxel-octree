#include "Quaternion.h"
#include "Mathlib.h"
#include "Mat4x4f.h"
#include "Vec3f.h"

namespace restless
{
	Quaternion::Quaternion() :
		W(1.0f), X(0.0f), Y(0.0f), Z(0.0f)
	{
	}

	Quaternion::Quaternion(const float w, const float x, const float y, const float z) :
		W(w), X(x), Y(y), Z(z)
	{
	}

	Quaternion::Quaternion(const float w, const Vec3f v):
		W(w), X(v[0]), Y(v[1]), Z(v[2])
	{
	}

	Quaternion operator * (const Quaternion &a, const Quaternion &b)
	{
		float w,x,y,z;

		w = a.W*b.W - (a.X*b.X + a.Y*b.Y + a.Z*b.Z);
  
		x = a.W*b.X + b.W*a.X + a.Y*b.Z - a.Z*b.Y;
		y = a.W*b.Y + b.W*a.Y + a.Z*b.X - a.X*b.Z;
		z = a.W*b.Z + b.W*a.Z + a.X*b.Y - a.Y*b.X;

		return Quaternion(w,x,y,z); 
	}

	const Quaternion& Quaternion::operator *= (const Quaternion &q)
	{
		float w = W*q.W - (X*q.X + Y*q.Y + Z*q.Z);

		float x = W*q.X + q.W*X + Y*q.Z - Z*q.Y;
		float y = W*q.Y + q.W*Y + Z*q.X - X*q.Z;
		float z = W*q.Z + q.W*Z + X*q.Y - Y*q.X;

		W = w;
		X = x;
		Y = y;
		Z = z;
		return *this;
	}

	const Quaternion& Quaternion::operator ~ ()
	{
		X = -X;
		Y = -Y;
		Z = -Z;
		return *this;
	}

	const Quaternion& Quaternion::operator - ()
	{
		float norme = sqrt(W*W + X*X + Y*Y + Z*Z);
		if (norme == 0.0f)
		norme = 1.0f;

		float recip = 1.0f / norme;

		W =  W * recip;
		X = -X * recip;
		Y = -Y * recip;
		Z = -Z * recip;

		return *this;
	}

	Quaternion Quaternion::conjugate() const
	{
		return Quaternion(W,-X,-Y,-Z);
	}

	void Quaternion::normalize()
	{
		float norme = sqrt(W*W + X*X + Y*Y + Z*Z);
		if (norme == 0.0) {
			W = 1.0; 
			X = 0.0f;
			Y = 0.0f;
			Z = 0.0f;
		} else {
			float recip = 1.0f/norme;

			W *= recip;
			X *= recip;
			Y *= recip;
			Z *= recip;
		}
	}

	Quaternion Quaternion::fromAxis(const float Angle, Vec3f v)
	{
		Quaternion qFromAxis;
		qFromAxis.X = 0.0f;
		qFromAxis.Y = 0.0f;
		qFromAxis.Z = 0.0f;
		qFromAxis.W = 1.0f;

		if (fabs(v.magnitude()) > FLT_EPSILON) {
			float omega= -0.5f * Angle;
			float c = (float)sin(omega);

			v = v.normalize();
			qFromAxis.X = c*v[0];
			qFromAxis.Y = c*v[1];
			qFromAxis.Z = c*v[2];
			qFromAxis.W = (float)cos(omega);
		} 

		qFromAxis.normalize();
		return qFromAxis;
	}

	Quaternion Quaternion::fromEuler(Vec3f v)
	{
		return fromEuler(v[0],v[1],v[2]);
	}

	Quaternion Quaternion::fromEuler(float x, float y, float z)
	{
		// taken from: http://www.gamedev.net/reference/articles/article1095.asp
		// a alternative solution could be: http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
		Quaternion Qx(cos(x/2), sin(x/2), 0, 0);
		Quaternion Qy(cos(y/2), 0, sin(y/2), 0);
		Quaternion Qz(cos(z/2), 0, 0, sin(z/2));

		Qx *= Qy*Qz;

		return Qx;
	}

	Quaternion Quaternion::interpolate(const Quaternion & a, const Quaternion & b, const float t)
	{
		Quaternion interpolated;
		interpolated.slerp(a, b, t);
		return interpolated;
	}

	/**
	 * Taken from the Assimp projects: aiQuaternion.h ... but to quote their code:
	 *
	 * " Performs a spherical interpolation between two quaternions 
	 *   Implementation adopted from the gmtl project. All others I found on the net fail in some cases.
	 *   Congrats, gmtl! "
	 *
	 */
	void Quaternion::slerp(const Quaternion &a,const Quaternion &b, const float t)
	{
		// calc cosine theta
		float cosom = a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;

		// adjust signs (if necessary)
		Quaternion end = b;

		if( cosom < 0.0f) {
			cosom = -cosom;
			end.X = -end.X;   // Reverse all signs
			end.Y = -end.Y;
			end.Z = -end.Z;
			end.W = -end.W;
		} 

		// Calculate coefficients
		float sclp, sclq;
		if((1.0f - cosom) > 0.0001f) { // 0.0001 -> some epsillon
			// Standard case (slerp)
			float omega, sinom;
			omega = acos( cosom); // extract theta from dot product's cos theta
			sinom = sin( omega);
			sclp  = sin( (1.0f - t) * omega) / sinom;
			sclq  = sin( t * omega) / sinom;
		} else {
			// Very close, do linear interp (because it's faster)
			sclp = 1.0f - t;
			sclq = t;
		}

		X = sclp * a.X + sclq * end.X;
		Y = sclp * a.Y + sclq * end.Y;
		Z = sclp * a.Z + sclq * end.Z;
		W = sclp * a.W + sclq * end.W;
	}

	Mat4x4f Quaternion::getMatrix() const
	{
		Mat4x4f matrix;

		matrix(0,0) = 1.0f - 2*Y*Y - 2*Z*Z;
		matrix(1,0) = 2*X*Y + 2*W*Z;
		matrix(2,0) = 2*X*Z - 2*W*Y;
		matrix(3,0) = 0.0f;

		matrix(0,1) = 2*X*Y - 2*W*Z;
		matrix(1,1) = 1.0f - 2*X*X - 2*Z*Z;
		matrix(2,1) = 2*Y*Z + 2*W*X;
		matrix(3,1) = 0.0f;

		matrix(0,2) = 2*X*Z + 2*W*Y;
		matrix(1,2) = 2*Y*Z - 2*W*X;
		matrix(2,2) = 1.0f - 2*X*X - 2*Y*Y;
		matrix(3,2) = 0.0f;

		matrix(0,3) = 0.0f;
		matrix(1,3) = 0.0f;
		matrix(2,3) = 0.0f;
		matrix(3,3) = 1.0f;

		return matrix;
	}

	const Quaternion& Quaternion::exp()
	{                               
		float Mul;
		float Length = sqrt(X*X + Y*Y + Z*Z);

		if (Length > 1.0e-4) {
			Mul = sin(Length)/Length;
		} else {
			Mul = 1.0;
		}
		
		W = cosf(Length);

		X *= Mul;
		Y *= Mul;
		Z *= Mul; 

		return *this;
	}

	const Quaternion& Quaternion::log()
	{
		float Length;

		Length = sqrt(X*X + Y*Y + Z*Z);
		Length = atan(Length/W);

		W = 0.0;

		X *= Length;
		Y *= Length;
		Z *= Length;

		return *this;
	}

	std::ostream & operator<<(std::ostream & os, const restless::Quaternion & q)	
	{
		os << std::setprecision(5) << std::fixed;
		os << "[";
		os << q.W << " " << q.X << " " << q.Y << " " << q.Z << " ";
		os << "]";
		return os;
	}
}