# OptNets_for_Shortest_Path_Interdiction


# Installation Instructions
Follow the below installation instructions to install the environment. Start by copying the github repository. This can be done with
>>

Next, we setup the local environment to run and modify the code.

Check if python is installed:
>> python3 --version
If no version is displayed, install python as instructed here: LINK NEEDED

Create and activate virtual environment:
>> python3 -m venv .venv
>> source ./.venv/bin/activate

Install all required libraries:
>> pip install requirements.txt

Run setup file:
>> pip install -e .

If you plan on making contributions run the following command to avoid loading notebook outputs to github:
>> nbstripout --install

