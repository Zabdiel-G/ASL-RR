import sys
import os
from asl_recognition.test import main as asl_rec_main
from chatbot.chatbot import main as chatbot_main
import subprocess

# Add the "src" directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

running=True
ans = 0 

while running:
    print ("""
    MENU
    
    1.Just Asl Recognition
    2.Just Chatbot
    3.Both
    4.Exit/Quit
    """)

    ans=input("What would you like to do?: ")
    if ans=="1": 
      print("\n Running ASL Recognition")
      asl_rec_main()
    elif ans=="2":
      print("\n Running Chatbot")
      chatbot_main()
    elif ans=="3":
      print("\n Running the Full program")
      print("\n 1st: Asl Recognition")
      asl_rec_main()
      print("\n 2nd: Chatbot")
      chatbot_main()
    elif ans=="4":
      print("\nGoodbye")
      running = False 
    elif ans !="":
      print("\n Not Valid Choice")
