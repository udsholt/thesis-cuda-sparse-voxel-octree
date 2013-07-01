#include "Core.h"

#include "GL.h"
#include <iostream>
#include <Mathlib/Vec2i.h>

#include "Engine/EngineInterface.h"
#include "Util/Screenshot.h"
#include "Util/Log.h"

namespace restless
{
	Core::Core() :
		ptrEngine(nullptr),
		_overlay(*this)
	{

	}

	Core::~Core()
	{

	}

	Input & Core::input()
	{
		return _input;
	}

	Timing & Core::timing()
	{
		return _timing;
	}

	Console & Core::console()
	{
		return _console;
	}

	VariableManager & Core::variables()
	{
		return _variables;
	}

	FileSystem & Core::files()
	{
		return FileSystem::getInstance();
	}

	OverlayManager & Core::overlay()
	{
		return _overlay;
	}

	void Core::parseArgs(const int argc, const char * argv[])
	{
		int i = 0;

		while (i < argc) {

			std::string arg = argv[i];
			i++;

			if (arg.compare("--data") == 0 && i < argc) {
				std::string dataDir = argv[i];
				i++;
				FileSystem::getInstance().setRoot(dataDir.c_str());
			}
			
		}

	}

	void Core::run(EngineInterface & engine)
	{
		ptrEngine = & engine;

		initialize();
		loop();
		shutdown();
	}

	void Core::initialize()
	{
		LOG(LOG_CORE, LOG_INFO) << "Filesystem: " << files().getRoot();

		// Initialize GLFW
		if(!glfwInit()) {
			return;
		}

		LOG(LOG_CORE, LOG_INFO) << "glfw initialized";

		EngineHints hints = ptrEngine->onRequestEngineHints();

		glfwOpenWindowHint(GLFW_WINDOW_NO_RESIZE, hints.windowAllowResize == false);

		glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, hints.openglVersionMajor);
		glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, hints.openglVersionMinor);
		glfwOpenWindowHint(GLFW_OPENGL_PROFILE, hints.openglProfile == EngineHints::OPENGL_PROFILE_CORE ? GLFW_OPENGL_CORE_PROFILE : GLFW_OPENGL_COMPAT_PROFILE);

		glfwOpenWindowHint(GLFW_FSAA_SAMPLES, hints.openglFFSAMultisamples);

		// Open an OpenGL window
		if(!glfwOpenWindow(hints.windowWidth, 
			               hints.windowHeight,
						   hints.openglRedBits,
						   hints.openglGreenBits,
						   hints.openglBlueBits,
						   hints.openglAlphaBits,
						   hints.openglDepthBits,
						   hints.openglStencilBits,
						   GLFW_WINDOW)) {
			glfwTerminate();
			return;
		}

		int major, minor, rev;
		glfwGetGLVersion(&major, &minor, &rev);
		LOG(LOG_CORE, LOG_INFO) << "glfw window opened";
		LOG(LOG_CORE, LOG_INFO) << "opengl version recieved: " << major << "." << minor << "." << rev;

		glfwSwapInterval(-1);

		// initialize GLEW
		// http://stackoverflow.com/questions/13558073/program-crash-on-glgenvertexarrays-call
		glewExperimental = GL_TRUE; 
		if( glewInit() != GLEW_OK ){
			LOG(LOG_CORE, LOG_ERROR) << "Failed to initialize GLEW";
			return;
		}

		glfwEnable( GLFW_KEY_REPEAT );

		glfwSetKeyCallback(__glfwKeyCallback);
		glfwSetCharCallback(__glfwCharCallback);
		glfwSetMouseButtonCallback(__glfwMouseButtonCallback);
		glfwSetWindowSizeCallback(__glfwWindowSizeCallback);

		// Setup console overlay
		_overlay.initialize();

		ptrEngine->initialize(); 
	}

	void Core::shutdown()
	{
		ptrEngine->shutdown(); 
		glfwTerminate();
	}

	void Core::loop()
	{
		running = true;

		while(running) {

			// Update the timings
			_timing.update();

			_input.update(_timing.getFrame());

			// Poll the mouse for position
			int x, y; glfwGetMousePos(&x, &y);
			_input.onMouseMove(Vec2i(x, y));

			ptrEngine->update();
			ptrEngine->render();

			// TODO: it might not be necessary to flush everytime
			if (_console.isEnabled()) {
				_console.flush(); 
			}

			// Update and render the overlay
			_overlay.update();
			_overlay.render();

			glfwSwapBuffers();
	
			// Handle window close
			running = running && glfwGetWindowParam( GLFW_OPENED );
		}
	}

	void Core::resizeWindow(const int width, const int height)
	{
		glfwSetWindowSize(width, height);
	}

	void Core::stop()
	{
		running = false;
	}

	void Core::onKey(const int key, const int action)
	{
		// activate console
		if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
			_console.toggleEnabled();
			return;
		}

		// disable stats overlay
		if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
			_overlay.toggleStats();
			return;
		}

		// take screenshot
		if(key == GLFW_KEY_F12 && action == GLFW_PRESS) {
			int width, height;
			glfwGetWindowSize(&width, &height);
			Screenshot::saveScreenshot(width, height);
			return;
		}

		if (_console.isEnabled() && action == GLFW_PRESS) {

			if (key == GLFW_KEY_ENTER) {
				_console.onSubmit();
				return;
			}

			if (key == GLFW_KEY_TAB) {
				_console.onComplete();
				return;
			}

			if (key == GLFW_KEY_UP) {
				_console.onHistoryNext();
				return;
			}

			if (key == GLFW_KEY_DOWN) {
				_console.onHistoryLast();
				return;
			}

			if (key == GLFW_KEY_LEFT) {
				_console.onCursorLeft();
				return;
			}

			if (key == GLFW_KEY_RIGHT) {
				_console.onCursorRight();
				return;
			}

			if (key == GLFW_KEY_BACKSPACE) {
				_console.onCharacterDelete();
				return;
			}

			if (key == GLFW_KEY_DEL) {
				_console.onLineDelete();
				return;
			}
		}

		_input.onKey(key, action);
		
	}

	void Core::onChar(const int character, const int action)
	{
		if (_console.isEnabled() && action == GLFW_PRESS) {
			_console.onCharacter((unsigned char) character);
		}
	}

	void Core::onMouseButton(const int button, const int action)
	{
		_input.onMouseButton(button, action);
	}

	void Core::onWindowResize(const int width, const int height)
	{
		ptrEngine->onWindowResize(width, height);
		_overlay.onViewportResize(width, height);
	}

	void __glfwKeyCallback(int key, int action)
	{
		Core::getInstance().onKey(key, action);
	}

	void __glfwMouseButtonCallback(int button, int action)
	{
		Core::getInstance().onMouseButton(button, action);
	}

	void __glfwWindowSizeCallback(int width, int height)
	{
		Core::getInstance().onWindowResize(width, height);
	}

	void __glfwCharCallback(int character, int action)
	{
		Core::getInstance().onChar(character, action);
	}

}
