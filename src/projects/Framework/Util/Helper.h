#ifndef _RESTLESS_UTIL_HELPER_H
#define _RESTLESS_UTIL_HELPER_H

#include "../GL.h"
#include <Mathlib/Mat4x4f.h>

namespace restless
{
	void clearGLError();
	bool logGLError(const char * what);
}


#endif