Work on this has temporarily slowed while I'm busy with university and various other things. 

A program to recognise birds seen on a live video feed. 

Uses vidgear (specifically camgear) to interpret youtube livestream and access frames. 
  I could've set up gstreamer to deal with freezing issue, which would allow me to use time.time() to get a frame each second, and run object detection on the same process. Instead, I used a counter to run it on every 30th frame (for 30fps stream) and multiprocessing (a new process the object detection function to get around the main process freezing). 

Model will learn from a tediously made database of hundreds of photos of birds per species implemented. Initially was using imageAI to make the model but I couldn't get it to work, so have switched to a more hands-on method with tensorflow. The visual recognition model is lacking in accuracy so most of my work will be outside of programming, learning how to get an effective NNet going. 

