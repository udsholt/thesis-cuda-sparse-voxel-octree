#ifndef _RESTLESS_VARIABLE_VARIBLEICONTAINER_H
#define _RESTLESS_VARIABLE_VARIBLEICONTAINER_H

#include "VariableInterface.h"
#include "VariableListener.h"

#include <string>
#include <sstream>

#include <Mathlib/Mathlib.h>
#include <Mathlib/Vec3f.h>

#include "../Util/StringTokenizer.h"
#include "../Util/Log.h"

namespace restless
{
	template<class T>
	class VariableContainer : public VariableInterface
	{
	public:
		VariableContainer(std::string name, T & variable) : 
			_name(name),
			_variable(variable),
			_ptrVariableListener(nullptr)
		{
			_orignalValue = _variable;
		};

		VariableContainer(std::string name, T & variable, VariableListener & listener) : 
			_name(name),
			_variable(variable),
			_ptrVariableListener(&listener)
		{
			_orignalValue = _variable;
		};

		virtual bool setValue(std::string & value)
		{
			if (internalSetValue(value)) {
				if (_ptrVariableListener != nullptr) {
					_ptrVariableListener->onVariableChange(_name.c_str());
				}
				return true;
			}

			return false;
		}

		virtual void resetValue()
		{
			_variable = _orignalValue;
			if (_ptrVariableListener != nullptr) {
				_ptrVariableListener->onVariableChange(_name.c_str());
			}
		}

		std::string getValue() const
		{
			std::stringstream stream;
			stream << _variable;
			return std::string(stream.str());
		}

		virtual std::string getName() const
		{
			return _name;
		}

		virtual std::string getTypeName() const
		{
			return "unknown";
		}
	
	protected:

		virtual bool internalSetValue(std::string & value)
		{
			return false;
		}

		T & _variable;
		T _orignalValue;
		std::string _name;

		VariableListener * _ptrVariableListener;
		std::string _listenerTag;
	};

	template <> std::string VariableContainer<bool>::getTypeName() const { return "bool"; }
	template <> std::string VariableContainer<int>::getTypeName() const { return "int"; }
	template <> std::string VariableContainer<unsigned int>::getTypeName() const { return "uint"; }
	template <> std::string VariableContainer<float>::getTypeName() const { return "float"; }
	template <> std::string VariableContainer<std::string>::getTypeName() const { return "string"; }
	template <> std::string VariableContainer<Vec3f>::getTypeName() const { return "vec3f"; }

	template <> 
	bool VariableContainer<bool>::internalSetValue(std::string & value) {  
		_variable = (value.compare("true") == 0 || value.compare("1") == 0) ;
		return true;
	}

	template <> 
	bool VariableContainer<int>::internalSetValue(std::string & value) { 
		_variable = atoi(value.c_str());
		return true;
	}

	template <> 
	bool VariableContainer<unsigned int>::internalSetValue(std::string & value) { 
		_variable = maxi(atoi(value.c_str()), 0);
		return true;
	}

	template <> 
	bool VariableContainer<float>::internalSetValue(std::string & value) { 
		_variable = ((float) atof(value.c_str()));
		return true;
	}

	template <> 
	bool VariableContainer<std::string>::internalSetValue(std::string & value) { 
		_variable = value;
		return true;
	}

	template <> 
	bool VariableContainer<Vec3f>::internalSetValue(std::string & value) {

		StringTokenizer tokens = StringTokenizer(value, " ");

		float x = tokens.nextFloatToken();
		float y = tokens.nextFloatToken();
		float z = tokens.nextFloatToken();

		_variable = Vec3f(x, y, z);

		return true;
	}

}

#endif