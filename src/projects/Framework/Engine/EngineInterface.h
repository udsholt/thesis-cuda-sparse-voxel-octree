#ifndef _RESTLESS_ENGINE_ENGINEINTERFACE_H
#define _RESTLESS_ENGINE_ENGINEINTERFACE_H

#include "EngineHints.h"

namespace restless
{
	class EngineInterface
	{
	public:
		virtual ~EngineInterface() {};

		

		virtual void initialize() = 0;
		virtual void shutdown() = 0;
		virtual void render() = 0;
		virtual void update() = 0;

		virtual void onWindowResize(const int width, const int height) = 0;

		virtual EngineHints onRequestEngineHints() = 0;
	};
}

#endif

