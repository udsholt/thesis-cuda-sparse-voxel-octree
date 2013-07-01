#ifndef _RESTLESS_UTIL_LOG_H
#define _RESTLESS_UTIL_LOG_H

// Highly inspired by: 
// http://www.drdobbs.com/cpp/201804215

#include <sstream>

#define LOG(channel, level) \
if (level > restless::Log::REPORTING_LEVEL) ; \
else restless::Log().get(channel, level, __FILE__, __LINE__)

#define L_INFO  LOG(restless::LOG_DEFAULT, restless::LOG_INFO)
#define L_DEBUG LOG(restless::LOG_DEFAULT, restless::LOG_DEBUG)
#define L_ERROR LOG(restless::LOG_DEFAULT, restless::LOG_ERROR)

namespace restless
{
	enum LOG_CHANNEL 
	{
		LOG_DEFAULT,
		LOG_CORE,
		LOG_RENDER,
		LOG_RESOURCE,
		LOG_INPUT,
		LOG_SHADER
	};

	enum LOG_LEVEL 
	{
		LOG_ERROR, 
		LOG_WARNING, 
		LOG_INFO, 
		LOG_DEBUG
	};

	class Log
	{
	private:
		Log(const Log &);
		Log & operator =(const Log&);

	protected:
		std::ostringstream os;

	public:
		Log();
		virtual ~Log();
		std::ostringstream & get(LOG_CHANNEL channel = LOG_DEFAULT, LOG_LEVEL level = LOG_INFO, const char * file = "UNKNOWN", int line = 0);

		static LOG_LEVEL REPORTING_LEVEL;
	};
}

#endif