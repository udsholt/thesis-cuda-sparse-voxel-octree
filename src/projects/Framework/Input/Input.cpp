#include "Input.h"

#include "../Util/Log.h"

using namespace std;

namespace restless
{
	Input::Input()
	{

	}


	Input::~Input()
	{

	}

	void Input::update(const unsigned int currentFrame)
	{
		frame = currentFrame;
	}

	void Input::mapButton(int key, const char * name)
	{
		keymap[key] = name;
	}

	void Input::mapMouseButton(int button, const char * name)
	{
		mousemap[button] = name;
	}

	const bool Input::getButton(const char * name)
	{
		string button = string(name);

		if (buttons.find(button) == buttons.end()) {
			return false;
		}

		return buttons[button];
	}

	const bool Input::getButtonDown(const char * name)
	{
		string button = string(name);

		if (buttonsDown.find(button) == buttonsDown.end()) {
			return false;
		}

		return frame - buttonsDown[button] <= 1;
	}

	void Input::onKey(const int key, const int action)
	{
		if (keymap.find(key) == keymap.end()) {
			return;
		}

		buttons[keymap[key]] = action == 1;

		if (action == 1) {
			buttonsDown[keymap[key]] = frame;
		}
	}

	void Input::onMouseButton(const int button, const int action)
	{
		if (mousemap.find(button) == mousemap.end()) {
			return;
		}

		buttons[mousemap[button]] = action == 1;
	}

	void Input::onMouseMove(const Vec2i position)
	{
		lastMousePosion = mousePosition;
		mousePosition = position;
	}

	const Vec2i Input::getMouseMovement() const
	{
		return lastMousePosion - mousePosition;
	}
}
