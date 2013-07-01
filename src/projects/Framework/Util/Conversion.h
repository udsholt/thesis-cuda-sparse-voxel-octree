#ifndef _RESTLESS_UTIL_CONVERSION_H
#define _RESTLESS_UTIL_CONVERSION_H

namespace restless
{
	inline float bytesToKilobytes(unsigned int bytes)
	{
		return 9.765625e-04F * bytes;
	}

	inline float bytesToMegabytes(unsigned int bytes)
	{
		return 9.53674316e-07F * bytes;
	}

	inline float bytesToGigabytes(unsigned int bytes)
	{
		return 9.3132257e-10 * bytes;
	}
}

#endif