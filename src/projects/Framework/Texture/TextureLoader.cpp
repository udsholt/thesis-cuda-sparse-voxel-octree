#include "TextureLoader.h"

#include <FreeImage.h>

#include "../GL.h"
#include "../Util/Log.h"

namespace restless
{
	Texture2D TextureLoader::load2DTexture(const char * filename, const bool flipVertical)
	{
		LOG(LOG_RESOURCE, LOG_INFO) << "loading texture from file: " << filename;

		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN; 
		FIBITMAP* dib(0); 

		// Check the file signature and deduce its format
		fif = FreeImage_GetFileType(filename, 0);

		// If still unknown, try to guess the file format from the file extension
		if(fif == FIF_UNKNOWN) {
			fif = FreeImage_GetFIFFromFilename(filename); 
		}

		// If still unkown, return failure
		if(fif == FIF_UNKNOWN) {
			LOG(LOG_RESOURCE, LOG_ERROR) << "unknown image format: " << filename;
			return Texture2D(); 
		}

		// Check if the plugin has reading capabilities and load the file
		if(FreeImage_FIFSupportsReading(fif)) {
			dib = FreeImage_Load(fif, filename); 
		}

		if(!dib) {
			LOG(LOG_RESOURCE, LOG_ERROR) << "free image does not support loading: " << filename;
			return Texture2D(); 
		}

		if (flipVertical) {
			FreeImage_FlipVertical(dib);
		}

		BYTE* bDataPointer = FreeImage_GetBits(dib); // Retrieve the image data

		unsigned int width = FreeImage_GetWidth(dib); // Get the image width and height
		unsigned int height = FreeImage_GetHeight(dib); 
		unsigned int bpp = FreeImage_GetBPP(dib); 
		unsigned int colorType = FreeImage_GetColorType(dib);

		// If somehow one of these failed (they shouldn't), return failure
		if(bDataPointer == NULL || width == 0 || height == 0) {
			LOG(LOG_RESOURCE, LOG_ERROR) << "load failed, no data: " << filename;
			return Texture2D();  
		}

		unsigned int textureId = 0;

		// Generate an OpenGL texture ID for this texture
		glGenTextures(1, & textureId); 
		glBindTexture(GL_TEXTURE_2D, textureId);

		// TODO: only handles GL_RGBA and GL_RGB
		int internalFormat = colorType == FIC_RGBALPHA ? GL_RGBA : GL_RGB;
		int pixelFormat = colorType == FIC_RGBALPHA ? GL_RGBA : GL_RGB;

		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, pixelFormat, GL_UNSIGNED_BYTE,(GLvoid*)bDataPointer );
		
		glGenerateMipmap(GL_TEXTURE_2D); 
		
		glBindTexture(GL_TEXTURE_2D, 0);

		FreeImage_Unload(dib); 

		return Texture2D(textureId, width, height);
	}
}