#include "Helper.h"

#include "Log.h"

namespace restless 
{
	void clearGLError()
	{
		glGetError();
	}

	bool logGLError(const char * what) 
	{
		GLenum errCode;
		const GLubyte *errString;

		if ((errCode = glGetError()) != GL_NO_ERROR) {
			errString = gluErrorString(errCode);
			LOG(LOG_RENDER, LOG_ERROR) << what << ", error: " << errString;
			return false;
		}

		return true;
	}

}