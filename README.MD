Project 2 - Python Team

- Might need to dropna() for the merge2 files

# Team Members and their contribution
Trevor Seibert - Project Leader
* Formated the project structure
* Spent Roughly 12hrs on Project

Crisitan Mayen - Researcher
* 
* 

Philip Ervin - Researcher 
* 
* 

Christopher Webster - Database Specialist 
* 
* 

# Dependencies 
pip install ~
* numpy
* pandas==1.3.5 (Only for profiling unless they made an update)
* pandas-profiling
* matplotlib
* scipy 
* PyQt5
* Tk (Tkinter)
* scikit-learn

# File Structure 
The_Code folder
* Inside the code folder, you will find all python files related to the steps in order. The main.py file only calls the code folder and python files. The reason being is that it will be easier to read for grading.

Log folder
* This is where the output which are recoreded via txt format. 

Profiling folder
* This is where you will find the summary statisics of the varibles. This is for step4-exploritory-linear-regression. Here you will find graphs from MatPlotLib and pandas profiling in HTML and PDF format. 

* On MAC/Linux, the file structure has to be / not \. Windows uses \ for the directories. 

# Git Good 

cd to where you want your project to be 

* git clone https://github.com/ts7even/eda-project2.git

## Git Push & Pull 
* git init
* git add . 
* git status (What branch you are on)
* git commit -m "First Commit"
* git push -u origin master 
Make sure you git pull before you start coding on other devives. 

## Create Git Branches for Team Members
* git status 'shows you what branch you are on'
* git branch (branch-name) (and or specific revision)
* git checkout (branch-name) - switches to diffrent branch
* git switch (branch-name - prefered way to switch to diffrent branch)
* git push -u origin (branch-name)

## Git Merge Merge Branches into master branch. Git rebase rewinds the latest merge
* git master (go into the master branch)
* git merge --squash feature (summarize all commits into one commit as latest commit into master branch)
* git commit -m "Branch merged into Master"
* git push 