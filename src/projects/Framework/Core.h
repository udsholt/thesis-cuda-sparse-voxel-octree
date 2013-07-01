#ifndef _RESTLESS_CORE_H
#define _RESTLESS_CORE_H

#include "Util/Singleton.h"
#include "Input/Input.h"
#include "Timing/Timing.h"
#include "Console/Console.h"
#include "Overlay/OverlayManager.h"
#include "Variable/VariableManager.h"
#include "Util/FileSystem.h"

namespace restless
{
	class EngineInterface;
}

namespace restless 
{
	class Core : public Singleton<Core>
	{
	protected:
		friend Singleton<Core>;

		Core();
		virtual ~Core();

	public:

		void parseArgs(const int argc, const char* argv[]);

		void run(EngineInterface & engine);
		void stop();

		void resizeWindow(const int width, const int height);

		Input & input();
		Timing & timing();
		Console & console();
		VariableManager & variables();
		FileSystem & files();
		OverlayManager & overlay();

		void onKey(const int key, const int action);
		void onChar(const int character, const int action);
		void onMouseButton(const int button, const int action);
		void onWindowResize(const int width, const int height);

	protected:

		EngineInterface * ptrEngine;

		Input _input;
		Timing _timing;
		Console _console;
		OverlayManager _overlay;
		VariableManager _variables;

		bool running;

		void initialize();
		void loop();
		void shutdown();
		
	};

	static void __glfwKeyCallback(int key, int action);
	static void __glfwCharCallback(int character, int action);
	static void __glfwMouseButtonCallback(int button, int action);
	static void __glfwWindowSizeCallback(int width, int height);
}

#endif