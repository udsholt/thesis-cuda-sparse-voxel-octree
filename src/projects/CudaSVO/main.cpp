#include <iostream>

#include "Engine.h"

int main( int argc, const char* argv[] )
{
	Engine engine = Engine();
	engine.preInitialize();
	engine.run(argc, argv);
}