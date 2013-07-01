#ifndef _RESTLESS_TEXTURE_TEXTETUREINTERFACE_H
#define _RESTLESS_TEXTURE_TEXTETUREINTERFACE_H

namespace restless
{
	class TextureInterface
	{
		virtual void bind(const unsigned int channel = 0) const = 0;
		virtual void unbind() const = 0;
	};
}

#endif