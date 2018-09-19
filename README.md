# choamodel

First, navigate to where you would like to place the folder on the command line (cd "blahblahblah/address/folderlocation")

Next, on the home page of the repo, click the "clone or download" button (green button in upper right) and copy the link.

Next, in the command line...

*********************************

*git clone [insert link here]*

*cd choamodel*

*********************************

Next, create your own branch (this is in case 2 or more ppl work on it at the same time- it ensures your work doesn't overwrite someone else's)

Type into command prompt:
*git checkout -b [name_of_your_new_branch]*

Change working branch :
*git checkout [name_of_your_new_branch]*

Push the branch on github :
*git push origin [name_of_your_new_branch]*

*********************************

If you want to add a completely new file to the repo:

drag the file from you local drive directly into the choamodel folder

then, in the command prompt:
*git add .* (this adds all new changes to an imaginary thingy)
then type:
*git commit -m "some commit message here like Add blah file"* (include the quotations)
then:
*git push origin [name_of_your_branch]*

and then finally:
*git checkout master*
*git merge [name_of_your_branch]* (merges new stuff from your branch to the master branch)

*********************************

*IMPORTANTE NOTE!* It's good practice to pull the latest version every time you work on this project. 
To do this:

*git checkout [name_of_your_branch]* (go to the master branch that's on your local machine)

*git pull origin master* (pulls files from remote master that's up in the github cloud into the version of master that's on your local machine)

then:
*git checkout [name_of_your_branch]* (go back to your branch)

optional but you should probably do this:
*git merge master* (merges the latest version of master into your branch)



