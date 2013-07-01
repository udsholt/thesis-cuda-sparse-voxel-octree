#ifndef _RESTLESS_VARIABLE_VARIABLEMANAGER_H
#define _RESTLESS_VARIABLE_VARIABLEMANAGER_H

#include "VariableInterface.h"
#include "VariableContainer.h"

#include <map>
#include <string>
#include <vector>

#include "VariableContainer.h"

namespace restless
{
	class VariableListener;
}

namespace restless
{

	class VariableManager
	{
	public:
		VariableManager();
		~VariableManager();

		template <class T>
		bool registerVariable(const char * name, T & variable)
		{
			return registerVariable<T>(std::string(name), variable);
		}

		template <class T>
		bool registerVariable(const std::string & name, T & variable)
		{
			if (exists(name)) {
				return false;
			}
			_variables.insert(std::pair<std::string,VariableInterface *>(std::string(name), new VariableContainer<T>(name, variable)));
			return true;
		}

		template <class T>
		bool registerVariable(const char * name, T & variable, VariableListener & listener)
		{
			return registerVariable<T>(std::string(name), variable, listener);
		}

		template <class T>
		bool registerVariable(const std::string & name, T & variable, VariableListener & listener)
		{
			if (exists(name)) {
				return false;
			}
			_variables.insert(std::pair<std::string,VariableInterface *>(std::string(name), new VariableContainer<T>(name, variable, listener)));
			return true;
		}

		

		bool exists(const char * name);
		bool exists(const std::string & name);

		VariableInterface & get(const char * name);
		VariableInterface & get(const std::string & name);

		std::vector<std::string> findByPrefix(const std::string & prefix);

	protected:

		int _nullInt;
		VariableContainer<int> _nullVariable;

		std::map<std::string, VariableInterface *> _variables;


	};

}

#endif