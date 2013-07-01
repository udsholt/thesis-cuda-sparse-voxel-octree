#ifndef _RESTLESS_INPUT_INPUT_H
#define _RESTLESS_INPUT_INPUT_H

#include <map>
#include <Mathlib/Vec2i.h>

namespace restless
{
	class Input
	{
	public:
		Input();
		~Input();

		void update(const unsigned int frame);

		void mapButton(int, const char *);
		void mapMouseButton(int, const char *);

		void onKey(const int, const int);
		void onMouseButton(const int, const int);
		void onMouseMove(const Vec2i);

		const Vec2i getMouseMovement() const;
		const bool getButton(const char *);
		const bool getButtonDown(const char *);

	protected:

		std::map<int, std::string> keymap;
		std::map<int, std::string> mousemap;

		std::map<std::string, bool> buttons;
		std::map<std::string, unsigned int>  buttonsDown;

		Vec2i mousePosition;
		Vec2i lastMousePosion;

		unsigned int frame;

	};
}

#endif
