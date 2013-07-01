#ifndef _RESTLESS_CONSOLE_CONSOLE_H
#define _RESTLESS_CONSOLE_CONSOLE_H

#include <string>
#include <sstream>
//#include <deque>
//#include <map>

#include "ConsoleHistory.h"
#include "ConsoleLog.h"
#include "ConsoleCommandManager.h"

namespace restless
{
	//class ConsoleReceiver;
	class ConsoleCommand;
}

namespace restless
{
	class Console
	{
	public:
		Console();
		virtual ~Console();

		bool registerCommand(const char * name, ConsoleCommand & command);

		std::string getLine() const;
		std::string getLog(const unsigned int lines) const;

		const unsigned int getLineTicks() const { return ticks; }
		const unsigned int getLogTicks() const { return log.getAdded(); }
		const unsigned int getCursorTicks() const { return cursor; }
		const unsigned int getCursor() { return cursor; }

		void onSubmit();
		void onComplete();
		void onHistoryNext();
		void onHistoryLast();
		void onCharacter(const char character);
		void onCharacterDelete();
		void onLineDelete();
		void onCursorLeft();
		void onCursorRight();
		void onCursorToEnd();

		const bool isEnabled() const;
		void setEnabled(bool enable);
		void toggleEnabled();

		std::ostringstream & out();
		void flush();

		void addLogline(const char * text);
		void addLogline(std::string text);

	protected:
		bool enabled;
		unsigned int ticks;
		unsigned int cursor;

		std::ostringstream stream;
		std::string line;

		ConsoleHistory history;
		ConsoleLog log;

		ConsoleCommandManager commands;
	};
}


#endif