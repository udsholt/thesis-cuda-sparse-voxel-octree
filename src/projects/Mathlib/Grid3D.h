#ifndef __RESTLESS_MATH_GRID3D_H
#define __RESTLESS_MATH_GRID3D_H

#include <iostream>

namespace restless
{
	template <class T> 
	class Grid3D
	{
		//std::vector<T> _data;

		// Stack allocated, consider implementing a allocator for this
		T * _data;

		unsigned int _width;
		unsigned int _height;
		unsigned int _depth;

	public:
		Grid3D(const unsigned int width, const unsigned int height, const unsigned int depth): 
			_width(width), 		
			_height(height),
			_depth(depth)
			//_data(width*height*depth)
		{
			_data = new T[_width*_height*_depth];	
		}

		~Grid3D()
		{
			delete[] _data;
		}

		const T * data() const 
		{
			return _data;
		}

		inline
		T operator()(const unsigned x, const unsigned y, const unsigned z) const
		{
			assert(inGrid(x,y,z));
			return _data[x + z*_width + y*_width*_depth];
		}

		inline
		T & operator()(const unsigned x, const unsigned y, const unsigned z)
		{
			assert(inGrid(x,y,z));
			return _data[x + z*_width + y*_width*_depth];
		}

		inline
		bool inGrid(const unsigned x, const unsigned y, const unsigned z) const {
			return x < _width && y < _height && z < _height;
		}
	
		/*
		void resize(const unsigned int width, const unsigned int height, const unsigned int depth)
		{
			_width = width;
			_height = height;
			_depth = depth;
			data.resize(_width*_depth*_height);
		}
		*/

		unsigned int width() const
		{
			return _width;
		}

		unsigned int height() const
		{
			return _height;
		}

		unsigned int depth() const
		{
			return _depth;
		}

		unsigned int size()
		{
			return _width * _height * _depth;
		}
	};

	template<class T> 
	std::ostream & operator<<(std::ostream & os, const restless::Grid3D<T> & grid)	
	{
		os << "[" << grid.width() << "x" << grid.height() << "x" << grid.depth() << "]";
		return os;
	}
};

#endif