#include "Settings.h"

#include <Framework/Variable/VariableManager.h>
#include <Framework/Variable/VariableListener.h>
#include <Framework/Util/Log.h>

#include "cuda/defines.h"

using namespace restless;

Settings::Settings() :
	debugEnableGizmos(true),
	raymarcherMaxDepth(MAX_DEPTH),
	animationPlaybackRate(0.1f),
	raymarchEnableRender(true),
	raymarcherEnableSubdivide(true),
	lightPosition(-10.0f, 10.0f, -10.0f),
	lightEnabled(true),
	raymarcherStepSize(1.0f),
	raymarcherRenderMode(0)
{
}

Settings::~Settings()
{
}

void Settings::initialize(VariableManager & variableManger, VariableListener & variableListener)
{
	variableManger.registerVariable<unsigned int>("raymarcher.max_depth", raymarcherMaxDepth);
	variableManger.registerVariable<bool>("raymarcher.enable_render", raymarchEnableRender);
	variableManger.registerVariable<bool>("raymarcher.enable_subdivide", raymarcherEnableSubdivide);
	variableManger.registerVariable<float>("raymarcher.step_size", raymarcherStepSize);
	variableManger.registerVariable<int>("raymarcher.render_mode", raymarcherRenderMode);
	variableManger.registerVariable<bool>("debug.enable_gizmos", debugEnableGizmos);
	variableManger.registerVariable<float>("animation.playback_rate", animationPlaybackRate);
	variableManger.registerVariable<Vec3f>("light.position", lightPosition);
	variableManger.registerVariable<bool>("light.enabled", lightEnabled);
}