# import asl_recognition.test
# from chatbot.chatbot import
import subprocess

running=True
ans = 0 

while running:
    print ("""
    1.Just Asl Recognition
    2.Just Chatbot
    3.Both
    4.Exit/Quit
    """)

    ans=input("What would you like to do?: ")
    if ans=="1": 
      print("\n Running ASL Recognition")
      subprocess.run(["python", "asl_recognition/test.py"]) 
    elif ans=="2":
      print("\n Running Chatbot") 
    elif ans=="3":
      print("\n Running the Full program") 
    elif ans=="4":
      print("\nGoodbye")
      running = False 
    elif ans !="":
      print("\n Not Valid Choice")
