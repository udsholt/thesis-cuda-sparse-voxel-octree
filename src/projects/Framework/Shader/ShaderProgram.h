#ifndef _RESTLESS_SHADER_SHADERPRORAM_H
#define _RESTLESS_SHADER_SHADERPRORAM_H

namespace restless
{
	class FileResource;
	class Vec2f;
	class Vec3f;
	class Vec2i;
	class Vec3i;
	class Vec4f;
	class Mat4x4f;
}

namespace restless
{
	class ShaderProgram
	{
	public:

		enum ShaderType
		{
			TYPE_VERTEX_SHADER,
			TYPE_FRAGMENT_SHADER
		};

		// http://www.opengl.org/discussion_boards/showthread.php/136657-glVertexAttribPointer-indices
		enum AttributeLocation
		{
			ATTRIB_VERTEX            = 0,
			ATTRIB_NORMAL            = 2,
			ATTRIB_COLOR             = 3,
			ATTRIB_SECONDARY_COLOR   = 4,
			ATTRIB_FOG_COORD         = 5,
			ATTRIB_MULTITEX_COORD_0  = 8,
			ATTRIB_MULTITEX_COORD_1  = 9,
			ATTRIB_MULTITEX_COORD_2  = 10,
			ATTRIB_MULTITEX_COORD_3  = 11, 
			ATTRIB_MULTITEX_COORD_4  = 12,
			ATTRIB_MULTITEX_COORD_5  = 13,
			ATTRIB_MULTITEX_COORD_6  = 14,
			ATTRIB_MULTITEX_COORD_7  = 15
		};

	public:
		ShaderProgram();
		~ShaderProgram();

		

		void unload();

		void enable();
		void disable();
		
		const unsigned int getProgramHandle();

		int getUniformLocation(const char * name) const;

		void setUniformBool(const char * name, bool val);
		void setUniformInt(const char * name, const int val);
		void setUniformFloat(const char * name, const float val);
		void setUniformVec2f(const char * name, const Vec2f & vector);
		void setUniformVec3f(const char * name, const Vec3f & vector);
		void setUniformVec2i(const char * name, const Vec2i & vector);
		void setUniformVec3i(const char * name, const Vec3i & vector);
		void setUniformVec4f(const char * name, const Vec4f & vector);
		void setUniformMatrix4x4f(const char * name, const Mat4x4f & matrix);

		int getAttributeLocation(const char * name) const;

		void enableAttributeArray(int attribute);

		bool addShaderFromFilename(ShaderType type, const char * filename);
		bool addShaderFromFileResource(ShaderType type, FileResource & file);
		bool addShaderFromSource(ShaderType type, const char * source);

		bool link();

	protected:
		unsigned int _programHandle;
		unsigned int _vertShaderHandle;
		unsigned int _fragShaderHandle;

		unsigned int compileShader(unsigned int type, const char * source);

		ShaderProgram(const ShaderProgram & other);
	};
}

#endif
