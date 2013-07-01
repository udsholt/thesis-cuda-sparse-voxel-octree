#ifndef _RESTLESS_UTIL_SINGLETON_H
#define _RESTLESS_UTIL_SINGLETON_H

namespace restless
{
	template <class T>
	class Singleton
	{
	public:
		static T & getInstance()
		{
			if (_singletonInstance == nullptr) {
				_singletonInstance = new T();
			}

			return *_singletonInstance;
		}

	protected:
		Singleton() {}
		virtual ~Singleton() {}

	private:
		static T * _singletonInstance;
	};

	template <typename T>
	T* Singleton<T>::_singletonInstance = nullptr;

}
#endif