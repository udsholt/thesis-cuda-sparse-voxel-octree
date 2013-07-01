#include "Console.h"

#include "../Util/Log.h"
#include "ConsoleCommand.h"
#include "ConsoleCommandRequest.h"

#include <vector>
#include <string>
#include <sstream>

using namespace std;

namespace restless
{
	Console::Console() :
		enabled(false),
		history(100),
		log(100),
		ticks(0),
		cursor(0)
	{
	}

	Console::~Console()
	{

	}


	bool Console::registerCommand(const char * name, ConsoleCommand & command)
	{
		return commands.registerCommand(name, command);
	}

	string Console::getLine() const
	{
		return line;
	}

	string Console::getLog(const unsigned int lines) const
	{
		vector<string> logLines = log.getLog(lines);

		stringstream logStream;
		
		if (lines > logLines.size()) {
			for (unsigned int i = 0; i < lines - logLines.size(); i++) {
				logStream << "\n";
			}
		}

		vector<string>::reverse_iterator  iter;
		for (iter = logLines.rbegin(); iter != logLines.rend(); iter++) {
			logStream << (*iter);
			if (iter + 1 != logLines.rend()) {
				logStream << "\n";
			}
		}

		return logStream.str();
	}

	void Console::onSubmit()
	{
		if (line.length() > 0) {
			history.add(line);

			commands.onExecute(ConsoleCommandRequest(line, out()));

			line = "";
			onCursorToEnd();
			ticks++;
		}
	}

	void Console::onComplete()
	{
		//if (std::string(line).length() > 0) {

			line = commands.onComplete(ConsoleCommandRequest(line, out()));
			onCursorToEnd();
			ticks++;
		//}
	}

	void Console::onCharacter(const char character)
	{
		line = line.insert(cursor, 1, character);
		cursor++;
		ticks++;
	}

	void Console::onCharacterDelete()
	{
		if (cursor < 1) {
			return;
		}

		if (line.length() > 0) {
			line = line.erase(cursor - 1, 1);
			cursor--;
			ticks++;
		}
	}

	void Console::onLineDelete()
	{
		line = "";
		onCursorToEnd();
		ticks++;
	}

	void Console::onCursorLeft()
	{
		if (cursor > 0) {
			cursor--;
			ticks++;
		}
	}

	void Console::onCursorRight()
	{
		if (cursor < line.length()) {
			cursor++;
			ticks++;
		}
	}

	void Console::onHistoryNext()
	{
		line = history.getNext();
		onCursorToEnd();
		ticks++;
	}

	void Console::onCursorToEnd()
	{
		cursor = line.length();
	}

	void Console::onHistoryLast()
	{
		line = history.getPrevious();
		cursor = line.length();
		ticks++;
	}

	const bool Console::isEnabled() const
	{
		return enabled;
	}

	void Console::setEnabled(bool enable)
	{
		enabled = enable;
	}

	void Console::toggleEnabled()
	{
		setEnabled(!isEnabled());
	}

	void Console::addLogline(const char * text)
	{
		addLogline(string(text));
	}

	void Console::addLogline(std::string text)
	{
		log.add(text);
	}

	ostringstream & Console::out()
	{
		return stream;
	}

	void Console::flush()
	{
		if (stream.str().size() == 0) {
			return;
		}

		// Get the contents of the stream and empty it
		string str = stream.str();
		stream.str("");

		// Was the buffer empty
		if (str.length() > 0) {

			// Loop while there are still newlines and add them to the log
			while(str.find("\n") != string::npos) {
				log.add( str.substr( 0, str.find("\n")) );
				str.erase(0, str.find("\n")+1);
			}
			
			// Add the remaining text if there is any
			if (str.length() > 0) log.add(str);
		}
	}

}