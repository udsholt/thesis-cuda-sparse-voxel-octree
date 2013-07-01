#ifndef _RESTLESS_SHADER_SHADERMANANAGER_H
#define _RESTLESS_SHADER_SHADERMANANAGER_H

#include <vector>
#include <string>

namespace restless
{
	class ShaderProgram;
}

namespace restless
{
	struct ShaderCacheHandle
	{
		std::string vertexShaderFilename;
		std::string fragmentShaderFilename;
		ShaderProgram * shaderProgram;
	};

	class ShaderManager
	{
	public:
		ShaderManager();
		~ShaderManager();

		ShaderProgram & loadShader(const char * vertexShaderFilename, const char * fragmentShaderFilename);
		void reloadShaders();

	protected:

		std::vector<ShaderCacheHandle> shaderProgramCache;

	};

}

#endif