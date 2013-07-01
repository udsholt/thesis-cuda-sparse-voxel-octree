#ifndef _RESTLESS_ENGINE_ENGINEBASIC_H
#define _RESTLESS_ENGINE_ENGINEBASIC_H

#include "EngineInterface.h"
#include "EngineHints.h"

#include "../Variable/Command/VariableSetCommand.h"
#include "../Variable/Command/VariableListCommand.h"
#include "../Variable/Command/VariableResetCommand.h"

namespace restless
{
	class Core;
}

namespace restless
{
	class EngineBasic : public EngineInterface
	{
	public:
		EngineBasic();
		virtual ~EngineBasic();

		virtual void run();
		virtual void run(const int argc, const char * argv[]);

		virtual void initialize();
		virtual void shutdown();
		virtual void render();
		virtual void update();

		virtual void onWindowResize(const int width, const int height);
		virtual EngineHints onRequestEngineHints();

	protected:

		Core & core;

		VariableSetCommand variableSetCommand;
		VariableListCommand variableListCommand;
		VariableResetCommand variableResetCommand;
	};
}

#endif
