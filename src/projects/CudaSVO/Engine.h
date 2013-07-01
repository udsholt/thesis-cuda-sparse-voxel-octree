#ifndef _INC_ENGINE_H
#define _InC_ENGINE_H

#include <Framework/Engine/EngineBasic.h>
#include <Framework/Shader/ShaderManager.h>
#include <Framework/Shader/ShaderProgram.h>
#include <Framework/Debug/DebugQuad.h>
#include <Framework/Debug/DebugCube.h>
#include <Framework/Debug/DebugGrid.h>
#include <Framework/Debug/DebugAxis.h>
#include <Framework/Variable/VariableListener.h>
#include <Framework/Camera/Camera.h>

#include <Framework/Animation/KeyframeAnimator.h>

#include <CudaFramework/CudaDevice.h>
#include <CudaFramework/CudaPixelBufferResource.h>
#include <CudaFramework/CudaTimerCollection.h>
#include <CudaFramework/CudaTimerCollectionRecorder.h>

#include "CameraKeyframe.h"
#include "CudaOctreeBuilder.h"
#include "Settings.h"
#include "CudaTimerCollectionOverlay.h"

class Engine : public restless::EngineBasic, public restless::VariableListener, public restless::ConsoleCommand
{
public:
	Engine();
	virtual ~Engine();

	virtual void preInitialize();
	virtual void initialize();
	virtual void shutdown();
	virtual void render();
	virtual void update();
	virtual void run(const int argc, const char * argv[]);

	virtual void onVariableChange(const char * name);

	virtual void onExecute(restless::ConsoleCommandRequest & request);
	virtual std::string onComplete(restless::ConsoleCommandRequest & request);

	virtual void onWindowResize(const int width, const int height);

	virtual restless::EngineHints Engine::onRequestEngineHints();

protected:

	CudaOctreeBuilder octreeBuilder;

	restless::CudaPixelBufferResource cudaPixelBuffer;
	restless::CudaDevice cudaDevice;
	
	restless::ShaderManager shaderManager;
	restless::ShaderProgram screenShader;
	restless::ShaderProgram debugShader;

	restless::DebugQuad screenQuad;
	restless::DebugGrid debugGrid;
	restless::DebugAxis debugAxis;
	restless::DebugCube debugCube;

	restless::KeyframeAnimator<CameraKeyframe> cameraAnimator;
	restless::Camera camera;

	Settings settings;

	restless::CudaTimerCollection cudaTimerCollection;
	restless::CudaTimerCollectionRecorder cudaTimerCollectionRecorder;

	CudaTimerCollectionOverlay cudaTimerOverlay;

	bool profileDelay;
	bool profileStopOnEend;
	float profileDelayTime;

	bool demo;
};

#endif

