# Accessing specific web pages in CARDS data server through SSH
The following tutorial can be followed irrespective of operating systems (Windows Subsystem for Linux (WSL) may need to be installed in older Windows versions).
- Download Firefox and install (if not installed already)
- Open command prompt/terminal and type the following:\\

ssh -N -D <port number> <username>@130.85.151.34 -p 2002

It will ask for user password and upon providing that it will remain in standby until you press ctrl-C.
- Open FireFox. Go to Settings> General > Network Settings

![plot](./fig/1.png)

- In the ‘Configure Proxy access to the Internet’ page follow the following images to setup (provide your chosen port number in the highlighted place). Click ‘Ok’ once done. cat

![plot](./fig/2.png)

![plot](./fig/3.png)

- To check whether the connection is successful or not search ‘what is my ip’ in google and you should get `130.85.151.34`
- Open a new tab in the browser and type the address of the webpage you want to visit

  ![plot](./fig/4.png)
