## Homework 2
Austin James
CSE 323
February 24, 2025

## Contents
 - task1.py
 - task2.py
 - task3.py

## Running each program
 - navigate to project directory
 - the python scripts are located in the src directory
 - their is a anaconda environment with the required packages
 - the program should pull your working directory for the image locations
 - if pulled from github the program expects these directories to save intermediate images
 -- output/task1
 -- output/task2
 -- output/task3

## observations
I found it quite difficult to get the image filtering to cooperate correctly, I found myself spending a lot of time just guessing and checking different values.  I found what worked best was firstly using a gaussian kernel for the closing and opening operations, the bell curve kernel helped remove noise the best.  One of the hardest things to do was find the correct balance of erosion to disconnect two objects but still keep enough of the object to be put back together and detected.  after spending a good amount of time playing with values for task2 I figured out that the best approach was to fill in the cheerios first and then erode them.  For task1 it was just about finding the best balance.  The biggest bug I had to deal with was the program was taking the background as an object, I was finally able to solve this when working on task3.  When I created the histogram I noticed that one value was super high which made me realize that it was the backgrounds perimeter and then I was able to solve that problem for both task1 and task3!