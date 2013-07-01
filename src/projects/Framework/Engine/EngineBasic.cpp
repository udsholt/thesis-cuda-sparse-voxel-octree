#include "EngineBasic.h"

#include "../Core.h"
#include "../GL.h"

namespace restless
{

	EngineBasic::EngineBasic() :
		core(Core::getInstance()),
		variableSetCommand(Core::getInstance().variables()),
		variableListCommand(Core::getInstance().variables()),
		variableResetCommand(Core::getInstance().variables())
	{

	}


	EngineBasic::~EngineBasic()
	{

	}

	void EngineBasic::run()
	{
		const int argc = 1;
		const char * argv[1] = { "foo" };
		run(argc, argv);
	}

	void EngineBasic::run(const int argc, const char * argv[])
	{
		core.parseArgs(argc, argv);
		core.run(*this);
	}

	void EngineBasic::initialize()
	{
		// Sane opengl setup
		glShadeModel(GL_SMOOTH);
		glClearColor(0.08f, 0.08f, 0.08f, 0.0f);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glEnable (GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// Commands setup
		core.console().registerCommand("set", variableSetCommand);
		core.console().registerCommand("list", variableListCommand);
		core.console().registerCommand("reset", variableResetCommand);
	}

	void EngineBasic::shutdown()
	{

	}

	void EngineBasic::render()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void EngineBasic::update()
	{

	}

	void EngineBasic::onWindowResize(const int width, const int height)
	{
		glViewport(0, 0, width, height);
	}

	EngineHints EngineBasic::onRequestEngineHints()
	{
		EngineHints hints = EngineHints();
		hints.windowWidth = 800;
		hints.windowHeight = 600;

		return hints;
	}

}