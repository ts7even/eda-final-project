Final Research Project - Python Team

# Team Members and their contribution
Trevor Seibert - Project Leader
* 
* 

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
* pandas
* pandas-profiling
* matplotlib
* scipy 
* PyQt5
* Tk (Tkinter)
* scikit-learn
* seaborn
* statsmodels

# File Structure 
The_Code folder
* Inside the code folder, you will find all python files for this project. The main.py file only calls the code folder and python files. 

Log folder
* This is where the output which are recoreded via txt format. 

Profiling folder
* This is where you will find the summary statisics of the varibles.

* On MAC/Linux, the file structure has to be / not \ :: Windows uses \ for the directories. 




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
* git merge --squash feature or git merge (branch Name) (summarize all commits into one commit as latest commit into master branch)
* git commit -m "Branch merged into Master"
* git push 

* git pull origin <branch>
