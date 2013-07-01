#include "Log.h"

namespace restless
{
	LOG_LEVEL Log::REPORTING_LEVEL = LOG_DEBUG;

	Log::Log()
	{

	}

	Log::~Log()
	{
		os << std::endl;
		fprintf(stderr, "%s", os.str().c_str());
		fflush(stderr);
	}

	std::ostringstream& Log::get(LOG_CHANNEL channel, LOG_LEVEL level, const char * file, int line)
	{
		std::string channelName = std::string("unknown");
		std::string levelName = std::string("unknown");

		switch(channel) {
			case LOG_DEFAULT:
				channelName = "deff";
				break;
			case LOG_CORE:
				channelName = "core";
				break;
			case LOG_RENDER:
				channelName = "rend";
				break;
			case LOG_RESOURCE:
				channelName = "ress";
				break;
			case LOG_INPUT:
				channelName = "inpt";
				break;
			case LOG_SHADER:
				channelName = "shad";
				break;
		}

		switch(level) {
			case LOG_ERROR:
				levelName = "error";
				break;
			case LOG_WARNING:
				levelName = "warn";
				break;
			case LOG_INFO:
				levelName = "info";
				break;
			case LOG_DEBUG:
				levelName = "debug";
				break;
		}

		os << "(" << channelName << ":" << levelName << ")\t";
		return os;
	}
}