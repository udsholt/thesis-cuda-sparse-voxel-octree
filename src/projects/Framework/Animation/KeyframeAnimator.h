#ifndef _RESTLESS_ANIMATION_KEYFRAMEANIMTOR_H
#define _RESTLESS_ANIMATION_KEYFRAMEANIMTOR_H

#include <vector>

namespace restless
{
	template<class T> 
	class KeyframeAnimator
	{
	public:

		KeyframeAnimator<T>() :
			playbackRate(1.0f),
			localTime(0.0f),
			playing(false)
		{
		}

		virtual ~KeyframeAnimator()
		{

		}

		bool isPlaying() const
		{
			return playing;
		}

		void setPlaying(const bool play)
		{
			playing = play;
		}

		void togglePlay()
		{
			playing = !playing;
		}

		void setPlaybackRate(const float & rate)
		{
			playbackRate = rate;
		}

		float getPlaybackRate() const
		{
			return playbackRate;
		}

		void reset()
		{
			localTime = 0.0f;
		}

		void update(const float delta)
		{
			if (!playing) {
				return;
			}

			float step = delta * playbackRate;

			if (localTime + step > lastKeyframe) {
				playing = false; // looping would be add and localTime -= lastKeyframe;
			}

			localTime += step;
		}

		T getAnimationKeyframe() const
		{
			unsigned int currIndex = (unsigned int) localTime;
			unsigned int nextIndex = currIndex + 1;

			if (nextIndex > lastKeyframe) {
				return keyframes[lastKeyframe];
			}

			float interp = localTime - currIndex;

			T current = keyframes[currIndex];
			T next = keyframes[nextIndex];

			return current.lerp(next, interp);
		}

		void addKeyframe(const T & keyframe)
		{
			keyframes.push_back(keyframe);
			lastKeyframe = keyframes.size() - 1;
		}

	protected: 

		bool playing;

		float playbackRate;
		float localTime;

		unsigned int lastKeyframe;

		std::vector<T> keyframes;

	};
}

#endif