#include "Engine.h"

// https://devtalk.nvidia.com/default/topic/468428/help-me-for-cudagraphicsglregisterimage/
// http://rauwendaal.net/2011/02/10/how-to-use-cuda-3-0s-new-graphics-interoperability-api-with-opengl/

#include <Mathlib/Vec2i.h>
#include <Mathlib/Vec2f.h>
#include <Mathlib/Mat4x4f.h>

#include <Framework/Core.h>
#include <Framework/Util/Log.h>
#include <Framework/GL.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <CudaFramework/CudaDevice.h>

#include "cuda/main.h"
#include "cuda/RenderInput.h"

using namespace restless;
using namespace std;

Engine::Engine() :
	cudaTimerOverlay(cudaTimerCollection),
	cudaTimerCollectionRecorder(cudaTimerCollection),
	profileDelay(false),
	profileStopOnEend(false)
{
}

Engine::~Engine()
{
}

EngineHints Engine::onRequestEngineHints()
{
	EngineHints hints       = EngineHints();
	hints.windowAllowResize = true;
	hints.windowWidth       = 800;
	hints.windowHeight      = 600;
	hints.openglProfile     = EngineHints::OPENGL_PROFILE_CORE;
	return hints;
}

void Engine::preInitialize()
{
	cudaDevice.listDevices();
	cudaDevice.chooseDevice(3, 0);
}

void Engine::run(const int argc, const char * argv[])
{
	EngineBasic::run(argc, argv);
}

void Engine::initialize()
{
	EngineBasic::initialize();

	// Camera setup
	camera.setPosition(Vec3f(2.0f, 2.0f, 2.0f));
	camera.rotate(2.4f, -0.5f);

	cameraAnimator.setPlaybackRate(0.5f);
	cameraAnimator.addKeyframe(CameraKeyframe(Vec3f(0.893694f, 0.680656f, 0.927389f), 2.61394f, 0.167331f));
	cameraAnimator.addKeyframe(CameraKeyframe(Vec3f(0.916888f, 0.686099f, 0.806141f), 2.1889f, 0.127331f));
	cameraAnimator.addKeyframe(CameraKeyframe(Vec3f(0.916888f, 0.686300f, 0.805141f), 2.1189f, 0.123331f));
	cameraAnimator.addKeyframe(CameraKeyframe(Vec3f(0.884348f, 0.703518f, 0.664927f), 1.76426f, 0.102938f));
	cameraAnimator.addKeyframe(CameraKeyframe(Vec3f(0.717133f, 0.731434f, 0.59688f), 1.92844f, 0.112961f));

	settings.animationPlaybackRate = cameraAnimator.getPlaybackRate();
	settings.initialize(core.variables(), *this);

	core.variables().registerVariable<bool>("timer.collect", cudaTimerCollection.enabled);
	core.variables().registerVariable<bool>("timer.show", cudaTimerOverlay.enabled);

	// Input setup
	core.input().mapButton(87, "foward"); 
	core.input().mapButton(65, "left");
	core.input().mapButton(83, "back");
	core.input().mapButton(68, "right");
	core.input().mapButton(GLFW_KEY_ESC, "escape");
	core.input().mapButton(GLFW_KEY_LSHIFT, "speed"); 
	core.input().mapButton(GLFW_KEY_F3, "toggleTimer"); 
	core.input().mapMouseButton(GLFW_MOUSE_BUTTON_1, "fire");

	// Shaders
	debugShader = shaderManager.loadShader(core.files().path("shaders/debug_color.vert").c_str(), core.files().path("shaders/debug_color.frag").c_str());
	screenShader = shaderManager.loadShader(core.files().path("shaders/screen_quad.vert").c_str(), core.files().path("shaders/screen_quad.frag").c_str());

	// Initialize the pixel buffer for the window
	cudaPixelBuffer.initialize(800, 600);

	// Setup a debug grid
	debugGrid.initialize(51, 0.5f);
	debugCube.initialize(Vec3f(0,0,0), Vec3f(1,1,1));
	debugAxis.initialize(51.0f * 0.5f);
	screenQuad.initialize();

	// Setup root for recorder
	cudaTimerCollectionRecorder.setDirectory(core.files().getRoot());

	// Finally console...
	core.console().registerCommand("exit", *this);
	core.console().registerCommand("record", *this);
	core.console().registerCommand("stop", *this);
	core.console().registerCommand("camera", *this);
	core.console().registerCommand("rebuild", *this);
	core.console().registerCommand("testpos", *this);
	core.console().registerCommand("animation", *this);
	core.console().registerCommand("profile", *this);
	core.console().registerCommand("resize_window", *this);
	core.console().out() << "F1 to toggles the console\n";
	core.console().out() << "F2 and F3 toggles stat-overlays\n";
	core.console().out() << "Press F12 to take a screenshot\n";
	core.console().out() << ".. tab for a list of console commands\n\n";
	
	core.console().setEnabled(true);


	octreeBuilder.initialize();


	cudaTimerCollection.initialize(BIN_LAST);
	cudaTimerCollection.registerBin(BIN_ALL,                             "ALL", true);
	cudaTimerCollection.registerBin(BIN_RENDER,                          "RENDER");
	cudaTimerCollection.registerBin(BIN_FILL_REQUEST_ARRAYS,             "FILL_REQUEST_ARRAYS");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_ALL,                   "SUBDIVIDE_ALL", true);
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_USAGE_MASK_TILE,       "SUBDIVIDE_USAGE_MASK_TILE");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_USAGE_MASK_BRICK,      "SUBDIVIDE_USAGE_MASK_BRICK");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_USAGE_MASK_COMPACT,    "SUBDIVIDE_USAGE_MASK_COMPACT");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_INVALIDATE_TIMESTAMPS, "SUBDIVIDE_INVALIDATE_TIMESTAMPS");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_INVALIDATE_NODES,      "SUBDIVIDE_INVALIDATE_NODES");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_REQUESTS_COMPACT,      "SUBDIVIDE_REQUESTS_COMPACT");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_REQUESTS_HANDLE,       "SUBDIVIDE_REQUESTS_HANDLE");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_REQUESTS_DATA,         "SUBDIVIDE_REQUESTS_DATA");
	cudaTimerCollection.registerBin(BIN_SUBDIVIDE_REQUESTS_FINALIZE,     "SUBDIVIDE_REQUESTS_FINALIZE");
	cudaTimerCollection.registerBin(BIN_COUNT_SUBDIVIDE_REQUEST,		 "COUNT_SUBDIVIDE_REQUEST");

	#ifdef KERNEL_TIMERS_ENABLE
		cudaTimerCollection.registerBin(BIN_KERNEL_ALL,					  "KERNEL_ALL", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_INNER,				  "KERNEL_INNER", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_RAYSETUP,		      "KERNEL_RAYSETUP", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_GLOBAL_INTERSECT,	  "KERNEL_GLOBAL_INTERSECT", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_LOOKUP,				  "KERNEL_LOOKUP", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_LOCALIZE,		      "KERNEL_LOCALIZE", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_LOCAL_INTERSECT,		  "KERNEL_LOCAL_INTERSECT", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_MARCH_CONSTANT,	      "KERNEL_MARCH_CONSTANT", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_MARCH_BRICK,			  "KERNEL_MARCH_BRICK", true);
		cudaTimerCollection.registerBin(BIN_KERNEL_SELECTED_SECTION,	  "KERNEL_SELECTED_SECTION", true);
	#endif

	cudaSetTimerCollection(cudaTimerCollection);

	core.overlay().addOverlay(cudaTimerOverlay);

}

void Engine::shutdown()
{

}

void Engine::update()
{
	float delta = core.timing().getTimeDelta();

	cameraAnimator.update(delta);

	// Stop on escape
	if (core.input().getButton("escape")) {
		core.stop();
	}
	
	if (!core.console().isEnabled()) {
		// Some deltas for moving around
		float moveSpeed = 0.1f * delta;
		float turnSpeed = 0.2f * delta;

		if (core.input().getButton("speed")) {
			moveSpeed *= 10.0f;
		}

		if (core.input().getButton("foward")) camera.forward(moveSpeed);
		if (core.input().getButton("back")) camera.forward(-moveSpeed);
		if (core.input().getButton("left")) camera.strafe(-moveSpeed);
		if (core.input().getButton("right")) camera.strafe(moveSpeed);

		if (core.input().getButton("fire")) {
			const Vec2i movement = core.input().getMouseMovement();
			if (movement[0] != 0 || movement[1] != 0)  {
				float pitch = - movement[0] * turnSpeed;
				float yaw = movement[1] * turnSpeed;
				camera.rotate(pitch, yaw);
			}
		}
	}

	if (core.input().getButtonDown("toggleTimer")) {
		cudaTimerCollection.enabled = !cudaTimerCollection.enabled;
		cudaTimerOverlay.enabled = cudaTimerCollection.enabled;
	}

	if (cameraAnimator.isPlaying()) {
		cameraAnimator.setPlaybackRate(settings.animationPlaybackRate);
		CameraKeyframe keyframe = cameraAnimator.getAnimationKeyframe();
		camera.setPosition(keyframe.position);
		camera.setRotation(keyframe.pitch, keyframe.yaw);
	}


	if (profileStopOnEend && cameraAnimator.isPlaying() == false) {
		profileStopOnEend = false;

		L_DEBUG << "STOPPING PROFILE";
		cudaTimerCollectionRecorder.stop();
	}


	if (profileDelay) {
			
		if (profileDelayTime > 0.0f) {
			L_DEBUG << "delay: " << profileDelayTime;
			profileDelayTime -= delta;	
		} else {
			profileDelay = false;
			profileStopOnEend = true;
			cameraAnimator.setPlaying(true);
			cudaTimerCollectionRecorder.start();
		}
	}

	

	cudaTimerCollectionRecorder.update(core.timing().getFPS());
}

void Engine::render()
{
	EngineBasic::render();

	if (settings.raymarchEnableRender) {
	
		float4 * devPtr = cudaPixelBuffer.mapDevicePointer();
		if (devPtr != nullptr) {

			RenderInput renderInput;

			renderInput.viewMatrixInverse    = make_matrix4x4(camera.getViewMatrixInverse().get());
			renderInput.viewFrustum          = make_frustum(camera.getFrustum());
			renderInput.buffer.width         = cudaPixelBuffer.getWidth();
			renderInput.buffer.height        = cudaPixelBuffer.getHeight();
			renderInput.buffer.ptr           = devPtr;
			renderInput.renderMode           = settings.raymarcherRenderMode;
			renderInput.maxDepth             = settings.raymarcherMaxDepth;
			renderInput.enableSubdivide      = settings.raymarcherEnableSubdivide;
			renderInput.stepSize             = settings.raymarcherStepSize / (BRICK_SIZE - 2 * BRICK_BORDER);
			renderInput.pixelSizeOnNearPlane = 1.0f / min(cudaPixelBuffer.getWidth(), cudaPixelBuffer.getHeight()); // 800x600 -> 0.001667
			renderInput.light.worldPosition  = make_float3(settings.lightPosition[0], settings.lightPosition[1], settings.lightPosition[2]);
			renderInput.light.enabled        = settings.lightEnabled;

			cudaVolumeRenderToPixelBuffer(renderInput, false);

			cudaPixelBuffer.unmapDevicePointer();
		}

	}

	if (settings.debugEnableGizmos) {
		debugShader.enable();
		debugShader.setUniformMatrix4x4f("projectionMatrix", camera.getProjectionMatrix());
		debugShader.setUniformMatrix4x4f("viewMatrix", camera.getViewMatrix());
		debugShader.setUniformVec4f("debugColor", Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
		debugAxis.draw();
		debugShader.setUniformVec4f("debugColor", Vec4f(0.25f, 0.25f, 0.25f, 1.0f));
		debugGrid.draw();
		debugShader.setUniformVec4f("debugColor", Vec4f(0.8f, 0.25f, 0.25f, 1.0f));
		debugCube.drawWire();
		debugShader.disable();
	}

	cudaPixelBuffer.bindBufferTexture(GL_TEXTURE0);
	screenShader.enable();
	screenShader.setUniformInt("buffer", 0);
	screenShader.setUniformVec2i("dimensions", Vec2i(cudaPixelBuffer.getWidth(), cudaPixelBuffer.getHeight()));
	screenShader.setUniformMatrix4x4f("projectionMatrix", Mat4x4f::ortho(0.0f, 1.0f, 0.0f, 1.0f, 0.1f, 100.0f));
	screenQuad.draw();
	screenShader.disable();
	cudaPixelBuffer.unbindBufferTexture();
}

void Engine::onVariableChange(const char * name)
{
}

void Engine::onWindowResize(const int width, const int height)
{
	EngineBasic::onWindowResize(width, height);
	camera.onViewportResize(width, height);
	cudaPixelBuffer.resize(width, height);
}

void Engine::onExecute(restless::ConsoleCommandRequest & request)
{
	if (request.command().compare("exit") == 0) {
		core.console().out() << "Bye bye!\n";
		core.stop();
		return;
	}

	if (request.command().compare("resize_window") == 0) {

		if (request.tokens().countTokens() < 2) {
			core.console().out() << "to few arguments for resize window\n";
			return;
		}

		const int width = request.tokens().nextIntToken();
		const int height = request.tokens().nextIntToken();

		if (width <= 50 || height <= 50) {
			core.console().out() << "window size to small min 50x50\n";
			return;
		}

		core.resizeWindow(width, height);
		core.console().out() << "Window resized to " << width << "x" << height << "!\n";
		return;
	}

	if (request.command().compare("record") == 0) {
		
		string filename = request.tokens().nextToken() + ".js";

		core.console().out() << "Recording to file: \"" << filename << "\"!\n";

		cudaTimerCollectionRecorder.setFilename(filename.c_str());

		if (!cudaTimerCollectionRecorder.start()) {
			core.console().out() << "....something went wrong!\n";
		}

		return;
	}

	if (request.command().compare("stop") == 0) {
		if (cudaTimerCollectionRecorder.stop()) {
			core.console().out() << "....recording stopped!\n";
			return;
		}

		core.console().out() << "....something went wrong!\n";
	}

	if (request.command().compare("camera") == 0) {
		core.console().out() << "Camera\n";
		core.console().out() << "  position: " << camera.getPosition() << "\n";
		core.console().out() << "  pitch:    " << camera.getPitch() << "\n";
		core.console().out() << "  yaw:      " << camera.getYaw() << "\n";
		core.console().out() << "\n";

		L_INFO << "cameraAnimator.addKeyframe(CameraKeyframe(Vec3f(" << camera.getPosition()[0] << "f, " << camera.getPosition()[1] << "f, " << camera.getPosition()[2]<< "f), " << camera.getPitch() << "f, " << camera.getYaw() << "f));";

		return;
	}

	if (request.command().compare("testpos") == 0) {
		camera.setPosition(Vec3f(0.222081f, 0.709071f, 0.789441f));
		camera.setRotation(3.81135f, 6.14626f);
		
		return;
	}

	if (request.command().compare("rebuild") == 0) {

		octreeBuilder.rebuild();

		return;
	}

	if (request.command().compare("profile") == 0) {

		ostringstream filename;
		filename << "profile" 
			     << "_rate-" << settings.animationPlaybackRate 
				 << "_maxdepth-" << settings.raymarcherMaxDepth 
				 << "_bricksize-" << BRICK_SIZE
				 << "_stepsize-" << settings.raymarcherStepSize
				 << "-timers-" << cudaTimerCollection.enabled
				 << ".js";

		L_INFO << "profiling to: " << filename.str();

		cudaTimerCollectionRecorder.setFilename(filename.str().c_str());

		cameraAnimator.reset();
		CameraKeyframe keyframe = cameraAnimator.getAnimationKeyframe();
		camera.setPosition(keyframe.position);
		camera.setRotation(keyframe.pitch, keyframe.yaw);

		octreeBuilder.rebuild();	

		profileDelay = true;
		profileStopOnEend = false;
		profileDelayTime = 2.0f;
		return;
	}

	if (request.command().compare("animation") == 0) {
		
		string argument = request.tokens().nextToken();

		if (argument.compare("play") == 0) {
			if (cameraAnimator.isPlaying()) {
				core.console().out() << "... animation is already playing\n";
				return;
			}
			cameraAnimator.reset();
			cameraAnimator.setPlaying(true);
			core.console().out() << "... playing animation\n";
			return;
		}

		if (argument.compare("stop") == 0) {
			if (!cameraAnimator.isPlaying()) {
				core.console().out() << "... animation is not playing\n";
				return;
			}
			cameraAnimator.setPlaying(false);
			core.console().out() << "... stopping animation\n";
			return;
		}

		if (argument.compare("reset") == 0) {
			cameraAnimator.reset();
			CameraKeyframe keyframe = cameraAnimator.getAnimationKeyframe();
			camera.setPosition(keyframe.position);
			camera.setRotation(keyframe.pitch, keyframe.yaw);

			core.console().out() << "... resetting animation\n";
			return;
		}

		core.console().out() << "unknown argument: " << argument << "\n";

		return;
	}
}

std::string Engine::onComplete(restless::ConsoleCommandRequest & request)
{
	if (request.command().compare("record") == 0) {
		return request.tokens().remainingString();
	}

	if (request.command().compare("animation") == 0) {
		string argument = request.tokens().nextToken();

		if (argument.substr(0, 1).compare("p") == 0) {
			return "play";
		}

		if (argument.substr(0, 1).compare("s") == 0) {
			return "stop";
		}

		if (argument.substr(0, 1).compare("r") == 0) {
			return "reset";
		}

		return " ";
	}

	return "";
}
