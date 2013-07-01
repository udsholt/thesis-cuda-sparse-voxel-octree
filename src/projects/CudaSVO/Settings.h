#ifndef _INC_SETTINGS_H
#define _INC_SETTINGS_H

#include <Mathlib/Vec3f.h>

namespace restless
{
	class VariableManager;
	class VariableListener;
}

class Settings
{
public:
	Settings();
	virtual ~Settings();

	void initialize(restless::VariableManager & variableManger, restless::VariableListener & variableListener);

	bool debugEnableGizmos;
	
	float animationPlaybackRate;

	bool raymarcherEnableSubdivide;

	bool raymarchEnableRender;
	unsigned int raymarcherMaxDepth;
	float raymarcherStepSize;
	int raymarcherRenderMode;

	restless::Vec3f lightPosition;
	bool lightEnabled;
};

#endif