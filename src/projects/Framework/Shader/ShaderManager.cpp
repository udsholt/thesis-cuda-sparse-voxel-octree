#include "ShaderManager.h"

#include "ShaderProgram.h"

#include "../Util/FileResource.h"
#include "../Util/Log.h"

using namespace std;

namespace restless
{

	ShaderManager::ShaderManager()
	{

	}


	ShaderManager::~ShaderManager()
	{
		for(vector<ShaderCacheHandle>::iterator iter = shaderProgramCache.begin(); iter != shaderProgramCache.end(); iter++) {
			delete iter->shaderProgram;
		}
	}

	ShaderProgram & ShaderManager::loadShader(const char * vertexShaderFilename, const char * fragmentShaderFilename)
	{
		// Has that shader already been loaded?
		for(vector<ShaderCacheHandle>::iterator iter = shaderProgramCache.begin(); iter != shaderProgramCache.end(); iter++) {
			if (iter->vertexShaderFilename.compare(vertexShaderFilename) == 0 && iter->fragmentShaderFilename.compare(fragmentShaderFilename) == 0) {
				LOG(LOG_RESOURCE, LOG_DEBUG) << "Returning cached shaderprogram: " << iter->shaderProgram->getProgramHandle();
				return *iter->shaderProgram;
			}
		}

		bool success = true;

		// Compile and link the shader program
		ShaderProgram * shaderProgram = new ShaderProgram();
		success = success && shaderProgram->addShaderFromFileResource(ShaderProgram::TYPE_VERTEX_SHADER, FileResource(vertexShaderFilename));
		success = success && shaderProgram->addShaderFromFileResource(ShaderProgram::TYPE_FRAGMENT_SHADER, FileResource(fragmentShaderFilename));
		success = success && shaderProgram->link();

		if (!success) {
			shaderProgram->unload();
		}

		ShaderCacheHandle cacheHandle;
		cacheHandle.vertexShaderFilename = vertexShaderFilename;
		cacheHandle.fragmentShaderFilename = fragmentShaderFilename;
		cacheHandle.shaderProgram = shaderProgram;
		shaderProgramCache.push_back(cacheHandle);

		return * shaderProgram;
	}

	void ShaderManager::reloadShaders()
	{
		LOG(LOG_RESOURCE, LOG_DEBUG) << "Reloading shaders";

		for(vector<ShaderCacheHandle>::iterator iter = shaderProgramCache.begin(); iter != shaderProgramCache.end(); iter++) {

			ShaderCacheHandle & cacheHandle = *iter;

			LOG(LOG_RESOURCE, LOG_DEBUG) << "reload vert: '" << cacheHandle.vertexShaderFilename << "' frag: '" << cacheHandle.fragmentShaderFilename << "'";

			cacheHandle.shaderProgram->unload();

			bool success = true;

			success = success && cacheHandle.shaderProgram->addShaderFromFileResource(ShaderProgram::TYPE_VERTEX_SHADER, FileResource(cacheHandle.vertexShaderFilename.c_str()));
			success = success && cacheHandle.shaderProgram->addShaderFromFileResource(ShaderProgram::TYPE_FRAGMENT_SHADER, FileResource(cacheHandle.fragmentShaderFilename.c_str()));
			success = success && cacheHandle.shaderProgram->link();

			if (!success) {
				cacheHandle.shaderProgram->unload();
			}
		}
		
	}

}