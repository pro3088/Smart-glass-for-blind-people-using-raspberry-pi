# Smart-glass-for-blind-people-using-raspberry-pi
## Brief Description
This approach examines the environment for potential threats along a walk path, including the distance to the object and the type of object, and notifies the user of the danger, allowing the user to move around freely without continual monitoring. After the model detects the object, an algorithm tracks the distance to the selected user and provides the user with the necessary information required to process his environment. 
The glass's camera collects data from the user's surroundings and delivers it to the model, which scans it and gives input on probable directions.

## Block Diagram of Proposed System
![image](https://user-images.githubusercontent.com/53413092/181853881-1f98663c-38bf-499e-ad36-4c379bff4a81.png)

## Flow chart of system
![image](https://user-images.githubusercontent.com/53413092/181854107-9257e672-8996-43de-9695-f496b11bc50f.png)

## Explanation of Working Concept (code)
### **CODE EXPLANATION**
This python script runs the main activity of the system. This code is majorly broken into three parts:

1.	Object detection
2.	Distance estimation
3.	Audio playback

In terms of object detection, the script uses a model that has previously been trained. This script employs the model to determine whether or not a known obstacle exists (cars and humans in this case). Because this system uses two cameras, detection must take place on both ends of the cameras. Threading is established as a notion that allows numerous processes to execute at the same time during code execution. The code has been separated into functions to make it easier to understand. Object detection is a function that contains all of the implementation for detecting recognized obstacles. The code is then executed for both threads, enabling detection for both cameras.

A function called depth is used to estimate distance. For compilation, this method accepts the left and right picture frames. The object detection function is then checked to see if an object has been detected. When an object is discovered, this function sends certain arguments to the triangulation script, which calculates and returns the estimated depth.

After the depth has been computed, another function called audio is called into action. This method takes the depth, converts it to a sentence, and then converts it to audio. The audio is then played for the blind to hear.


## Implementation of Design
![image](https://user-images.githubusercontent.com/53413092/181854175-59785071-797a-413d-a7bc-101c7acd7c1e.png)

## Prove of working Concept
![image](https://user-images.githubusercontent.com/53413092/181854216-0594bae3-e7af-4091-b223-87644b84e779.png)
![image](https://user-images.githubusercontent.com/53413092/181854279-e1316bae-ccb4-423e-a2d2-8322ce607b4c.png)

## Estimated Cost of project

### Note:
This cost was taken at March, 2022 in the nigerian local currency

| S/No| Components| Quantity | Unit Price (N)	 | Total Price (N) |
| :---: | :----------------: | :--------: | :---------------: | :---------------: |
| 1 | Raspberry Pi – 2gb | 1 | 	N 34,000 | 	N34,000 |
| 2 |	8MP USB Camera |	2	 | N 17,000	| N 34,000 |
| 4	| Vero Board | 1 |	N 250 |	N 250 |
| 5	| 4- piece raspberry pi heat sink |	1 |	N 800 |	N 800 |
| 6	| Raspberry pi case |	1 |	N 2500 | N 2500 |
| 7	| Lipo 18650 Battery |	2 |	1500 |	3000 |
| 8	| 18650 1S Lipo battery holder |	2	| 300	| 600 |
| 9	| Miscellaneous (shipping and others) |	| |	N22,000 |
| TOTAL| 	| | |	N 97,150 |

### **note**
The raspberry pi isnt the best hardware to run this project as it doesn't have TPU, some processes couldn't be tested and the hardware runs slowly.

This is just to show the potential of using a more suitable hardware and running the processes on it.

note that some edits will have to be done to make the code efficient on a better device (hardware). i.e making the two camera's read distance better and more optimally.

You can read more from: https://www.researchgate.net/publication/362345190_DESIGN_AND_IMPLEMENTATION_OF_SMART_GLASSES_FOR_BLIND_PEOPLE_-_Using_Raspberry_PI

Thank you.

