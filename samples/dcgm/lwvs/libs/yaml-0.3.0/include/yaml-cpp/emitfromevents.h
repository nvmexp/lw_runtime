#ifndef EMITFROMEVENTS_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define EMITFROMEVENTS_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) || (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || (__GNUC__ >= 4)) // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/eventhandler.h"
#include <stack>

namespace YAML
{
	class Emitter;
	
	class EmitFromEvents: public EventHandler
	{
	public:
		EmitFromEvents(Emitter& emitter);
		
		virtual void OnDolwmentStart(const Mark& mark);
		virtual void OnDolwmentEnd();
		
		virtual void OnNull(const Mark& mark, anchor_t anchor);
		virtual void OnAlias(const Mark& mark, anchor_t anchor);
		virtual void OnScalar(const Mark& mark, const std::string& tag, anchor_t anchor, const std::string& value);
		
		virtual void OnSequenceStart(const Mark& mark, const std::string& tag, anchor_t anchor);
		virtual void OnSequenceEnd();
		
		virtual void OnMapStart(const Mark& mark, const std::string& tag, anchor_t anchor);
		virtual void OnMapEnd();
		
	private:
		void BeginNode();
		void EmitProps(const std::string& tag, anchor_t anchor);
		
	private:
		Emitter& m_emitter;
		
		struct State { enum value { WaitingForSequenceEntry, WaitingForKey, WaitingForValue }; };
		std::stack<State::value> m_stateStack;
	};
}

#endif // EMITFROMEVENTS_H_62B23520_7C8E_11DE_8A39_0800200C9A66
