ffmpeg -i raindrops-2.mp4 -s 256x256 -filter:v "setpts=0.5*PTS" -sws_flags neighbor raindrops-2-scaled-fast.gif
