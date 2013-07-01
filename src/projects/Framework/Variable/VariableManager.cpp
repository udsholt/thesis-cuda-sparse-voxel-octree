#include "VariableManager.h"

#include "../Util/Log.h"

using namespace std;

namespace restless
{
	VariableManager::VariableManager(void) :
		_nullVariable("nullvariable", _nullInt)
	{
	}


	VariableManager::~VariableManager(void)
	{
	}

	bool VariableManager::exists(const char * name)
	{
		return exists(string(name));
	}

	bool VariableManager::exists(const string & name)
	{
		return _variables.find(name) != _variables.end();
	}

	VariableInterface & VariableManager::get(const char * name) 
	{
		return get(string(name));
	}

	VariableInterface & VariableManager::get(const string & name) 
	{
		map<string,VariableInterface *>::iterator it = _variables.find(string(name));
		if (it == _variables.end()) {
			return _nullVariable;
		}
		return * it->second;
	}

	vector<string> VariableManager::findByPrefix(const std::string & prefix)
	{
		vector<string> matches = vector<string>();

		map<string,VariableInterface *>::iterator it;
		for(it = _variables.begin(); it != _variables.end(); it++) {

			const string name = it->first;

			if (name.length() < prefix.length()) {
				continue;
			}

			if (name.substr(0, prefix.length()).compare(prefix) == 0) {
				matches.push_back(name);
			}
		}

		return matches;
	}

}