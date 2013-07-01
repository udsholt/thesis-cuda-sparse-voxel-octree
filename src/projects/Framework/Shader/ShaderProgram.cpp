#include "ShaderProgram.h"

#include "../GL.h"

#include "../Util/FileResource.h"
#include "../Util/Log.h"

#include <Mathlib/Vec2i.h>
#include <Mathlib/Vec2f.h>
#include <Mathlib/Vec3f.h>
#include <Mathlib/Vec3i.h>
#include <Mathlib/Mat4x4f.h>

namespace restless 
{

	ShaderProgram::ShaderProgram() :
		_programHandle(0),
		_vertShaderHandle(0),
		_fragShaderHandle(0)
	{

	}

	ShaderProgram::ShaderProgram(const ShaderProgram & other)
	{
	}


	ShaderProgram::~ShaderProgram()
	{

	}

	void ShaderProgram::unload()
	{
		// Delete the program, this also detaches any shaders
		if (_programHandle != 0) glDeleteProgram(_programHandle);

		// Delete shaders
		if (_vertShaderHandle != 0) glDeleteShader(_vertShaderHandle);
		if (_fragShaderHandle != 0) glDeleteShader(_fragShaderHandle);

		_programHandle    = 0;
		_vertShaderHandle = 0;
		_fragShaderHandle = 0;
	}

	void ShaderProgram::enable()
	{
		glUseProgram(_programHandle);
	}

	void ShaderProgram::disable()
	{
		glUseProgram(0);
	}

	const unsigned int ShaderProgram::getProgramHandle()
	{
		return _programHandle;
	}

	void ShaderProgram::setUniformVec2i(const char * name, const Vec2i & vector)
	{
		glUniform2iv(getUniformLocation(name), 1, vector.get());
	}

	void ShaderProgram::setUniformVec3i(const char * name, const Vec3i & vector)
	{
		glUniform3iv(getUniformLocation(name), 1, vector.get());
	}

	void ShaderProgram::setUniformVec3f(const char * name, const Vec3f & vector)
	{
		glUniform3fv(getUniformLocation(name), 1, vector.get());
	}

	void ShaderProgram::setUniformVec2f(const char * name, const Vec2f & vector)
	{
		glUniform2fv(getUniformLocation(name), 1, vector.get());
	}

	void ShaderProgram::setUniformVec4f(const char * name, const Vec4f & vector)
	{
		glUniform4fv(getUniformLocation(name), 1, vector.get());
	}

	void ShaderProgram::setUniformMatrix4x4f(const char * name, const Mat4x4f & matrix)
	{
		glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, matrix.get());
	}

	void ShaderProgram::setUniformBool(const char * name, bool val)
	{
		glUniform1i(getUniformLocation(name), val);
	}

	void ShaderProgram::setUniformInt(const char * name, int val)
	{
		glUniform1i(getUniformLocation(name), val);
	}

	void ShaderProgram::setUniformFloat(const char * name, float val)
	{
		glUniform1f(getUniformLocation(name), val);
	}

	int ShaderProgram::getUniformLocation(const char * name) const
	{
		return glGetUniformLocation(_programHandle, name);
	}

	int ShaderProgram::getAttributeLocation(const char * name) const
	{
		return glGetAttribLocation(_programHandle, name);
	}

	void ShaderProgram::enableAttributeArray(int attribute)
	{
		glEnableVertexAttribArray(attribute);
	}

	bool ShaderProgram::link()
	{
		if (_vertShaderHandle == 0 && _fragShaderHandle == 0) {
			LOG(LOG_SHADER, LOG_ERROR) << "Link failed: missing vertex or fragment shader";
			return false;
		}

		// Create a new shader program
		unsigned int prg = glCreateProgram();	
		
		// Attach the two shaders we have compiled
		glAttachShader(prg, _vertShaderHandle);
		glAttachShader(prg, _fragShaderHandle);

		// TODO: this should not be hardcoded
		glBindAttribLocation(prg, ATTRIB_VERTEX, "vertexPosition");
		glBindAttribLocation(prg, ATTRIB_COLOR, "vertexColor");
		glBindAttribLocation(prg, ATTRIB_MULTITEX_COORD_0, "vertexTexcoord");
		
		// Link the program
		int linked = GL_FALSE;
		glLinkProgram(prg);	
		glGetProgramiv(prg, GL_LINK_STATUS, &linked); 
		
		// On error we display a log message
		if (linked == GL_FALSE){
			
			// Get the lenght of the error message
			int len = 0;
			glGetProgramiv(prg, GL_INFO_LOG_LENGTH, &len);

			// Get the log message
			char* log = (char*)malloc(len);					// TODO: Memory
			glGetProgramInfoLog(prg, len, NULL, log); 

			// Display the error
			LOG(LOG_SHADER, LOG_ERROR) << "Link failed:\n\n" << log;

			// Clear the log
			free(log);

			// Delete the program we created
			glDeleteProgram(prg);
			
			return false;
		}

		_programHandle = prg;

		return true;
	}

	bool ShaderProgram::addShaderFromFilename(ShaderType type, const char * filename)
	{
		return addShaderFromFileResource(type, FileResource(filename));
	}

	bool ShaderProgram::addShaderFromFileResource(ShaderProgram::ShaderType type, FileResource & file)
	{
		LOG(LOG_SHADER, LOG_INFO) << "Loading shader from file: " << file.getFilename();
		return addShaderFromSource(type, file.getContents());
	}

	bool ShaderProgram::addShaderFromSource(ShaderProgram::ShaderType type, const char * source)
	{
		if (type == TYPE_VERTEX_SHADER) {
			_vertShaderHandle = compileShader(GL_VERTEX_SHADER, source);
			return true;
		}
		if (type == TYPE_FRAGMENT_SHADER) {
			_fragShaderHandle = compileShader(GL_FRAGMENT_SHADER, source);
			return true;
		}

		return false;
	}

	unsigned int ShaderProgram::compileShader(unsigned int type, const char * source)
	{
		// Create a new shader handle
		unsigned int handle = glCreateShader(type);

		// Compile the shader and get the result
		int compiled = GL_FALSE;
		glShaderSource(handle, 1, &source, NULL);
		glCompileShader(handle);	
		glGetShaderiv(handle, GL_COMPILE_STATUS, &compiled); 
		
		// If the load has failed we output the result in the log
		if (compiled == GL_FALSE){
			
			// Get the lenght of the error message
			int len = 0;
			glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &len);

			// Get the log message
			char* log = (char*)malloc(len);
			glGetShaderInfoLog(handle, len, NULL, log); 

			// Display the error
			LOG(LOG_SHADER, LOG_ERROR) << "Compile failed: " << (type == GL_VERTEX_SHADER ? "GL_VERTEX_SHADER" : "GL_FRAGMENT_SHADER") << "\n\n" << log;

			// Clear the log
			free(log);

			// Destroy the shader again
			glDeleteShader(handle);

			return 0;
		}

		return handle;
	}

	
}
