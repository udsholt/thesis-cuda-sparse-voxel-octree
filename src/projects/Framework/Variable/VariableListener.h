#ifndef _RESTLESS_VARIABLE_VARIBLELISTENER_H
#define _RESTLESS_VARIABLE_VARIBLELISTENER_H

namespace restless
{
	class VariableListener
	{
	public:
		virtual void onVariableChange(const char * name) = 0;
	};
}

#endif