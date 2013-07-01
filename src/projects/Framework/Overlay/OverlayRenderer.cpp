#include "OverlayRenderer.h"

#include "../GL.h"
#include "../Util/Log.h"
#include "../Shader/ShaderProgram.h"
#include "../Util/FileResource.h"
#include "../Debug/DebugQuad.h"

#include "TextObject.h"
#include "TextFont.h"

#include <Mathlib/Vec3f.h>
#include <Mathlib/Vec2f.h>

#include "../Util/FileSystem.h"

#include <string>

using namespace std;

namespace restless
{

	OverlayRenderer::OverlayRenderer() :
		textShader(),
		primitveShader(),
		projection(Mat4x4f::ortho(0, 640, 480, 0, 0.5, 100))
	{
	}


	OverlayRenderer::~OverlayRenderer()
	{
	}

	void OverlayRenderer::onViewportResize(const int width, const int height)
	{
		projection = Mat4x4f::ortho(0, (float) width, (float) height, 0, 0.5, 100);
	}

	void OverlayRenderer::initialize()
	{
		FileSystem & files = FileSystem::getInstance();

		textShader.addShaderFromFileResource(ShaderProgram::TYPE_VERTEX_SHADER, FileResource(files.path("shaders/overlay/text.vert").c_str()));
		textShader.addShaderFromFileResource(ShaderProgram::TYPE_FRAGMENT_SHADER, FileResource(files.path("shaders/overlay/text.frag").c_str()));
		textShader.link();

		primitveShader.addShaderFromFileResource(ShaderProgram::TYPE_VERTEX_SHADER, FileResource(files.path("shaders/overlay/primitive.vert").c_str()));
		primitveShader.addShaderFromFileResource(ShaderProgram::TYPE_FRAGMENT_SHADER, FileResource(files.path("shaders/overlay/primitive.frag").c_str()));
		primitveShader.link();
	}

	void OverlayRenderer::drawText(const TextObject & text)
	{
		glDisable(GL_DEPTH_TEST);

		text.getFont().getTexture().bind(0);

		textShader.enable();
		textShader.setUniformVec2f("offset", text.getPosition());
		textShader.setUniformVec4f("tint", text.getColor());
		textShader.setUniformInt("overlayTexture", 0);
		textShader.setUniformMatrix4x4f("projectionMatrix", projection);

		text.draw();

		textShader.disable();
		
		text.getFont().getTexture().unbind();

		glEnable(GL_DEPTH_TEST);


	}

	void OverlayRenderer::drawQuad(const DebugQuad & quad, const Vec4f color)
	{
		glDisable(GL_DEPTH_TEST);

		primitveShader.enable();
		primitveShader.setUniformVec4f("tint", color);
		primitveShader.setUniformFloat("overlayTextureContribution", 0.0f);
		primitveShader.setUniformMatrix4x4f("projectionMatrix", projection);

		quad.draw();

		primitveShader.disable();

		glEnable(GL_DEPTH_TEST);
	}

}