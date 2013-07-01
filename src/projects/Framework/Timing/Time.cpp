#include "Time.h"

// Only need the timer from glfw, but glew must always be included first
#include "../GL.h"

namespace restless
{
	double Time::getTime()
	{
		return glfwGetTime();
	}
}