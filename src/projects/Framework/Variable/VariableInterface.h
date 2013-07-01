#ifndef _RESTLESS_VARIABLE_VARIBLEINTERFACE_H
#define _RESTLESS_VARIABLE_VARIBLEINTERFACE_H

#include <string>

namespace restless
{
	class VariableInterface
	{
	public:
		
		virtual bool setValue(std::string & value) = 0;
		virtual bool setValue(const char * value) { return setValue(std::string(value)); };

		virtual void resetValue() = 0;

		virtual std::string getValue() const = 0;

		virtual std::string getName() const = 0;

		virtual std::string getTypeName() const = 0;
	};
}

#endif

